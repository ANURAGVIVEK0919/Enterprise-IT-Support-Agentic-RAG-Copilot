import logging
import re
import secrets
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents 
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
settings = get_settings()

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question)
        write_audit(payload.question, result["source_used"], result.get("trace", []))

        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", payload.question),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error processing chat question: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your request. Please try again later."
        ) from exc


@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    if not secrets.compare_digest(x_admin_key.strip(), settings.admin_api_key.strip()):
        raise HTTPException(status_code=401, detail="Invalid admin key")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported file formats: {', '.join(sorted(SUPPORTED))}")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    raw_name = Path(file.filename or "upload").name
    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_name).strip("._")
    if not clean_name or not clean_name.endswith(suffix):
        clean_name = f"document{suffix}"
    dest = upload_dir / clean_name

    # Stream file to disk enforcing max upload size
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    size = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum allowed size of {settings.max_upload_size_mb} MB."
                    )
                f.write(chunk)

        # Offload file parsing, chunking, and embedding to worker threads
        docs = await run_in_threadpool(load_file, dest)
        chunks = await run_in_threadpool(chunk_documents, docs)
        ids = await run_in_threadpool(add_documents, chunks)

        return {
            "message": "Document indexed",
            "file": dest.name,
            "chunks": len(chunks),
            "ids_created": len(ids),
        }
    except HTTPException:
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.exception("Error during document ingestion: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while parsing and indexing the document."
        ) from exc
