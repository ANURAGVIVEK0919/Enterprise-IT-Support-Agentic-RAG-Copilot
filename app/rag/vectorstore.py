import time
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import get_settings

settings = get_settings()


_embeddings = None
_vectorstore = None


EMBEDDING_DIMENSIONS = {
    "models/gemini-embedding-001": 768,
    "gemini-embedding-001": 768,
    "models/gemini-embedding-2": 768,
    "gemini-embedding-2": 768,
    "models/text-embedding-004": 768,
    "text-embedding-004": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-minilm-l6-v2": 384,
}



def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()
    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()
    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]
    if "gemini-embedding" in normalized:
        return 768
    if "text-embedding-004" in normalized:
        return 768
    if "text-embedding-3-small" in normalized:
        return 1536
    if "text-embedding-3-large" in normalized:
        return 3072
    if "text-embedding-ada-002" in normalized:
        return 1536
    if "all-minilm" in normalized:
        return 384

    raise ValueError(
        f"Unsupported embedding model '{model_name or settings.embedding_model}' for Pinecone. "
        "Add the matching dimension to EMBEDDING_DIMENSIONS."
    )



def get_embeddings():
    global _embeddings
    if _embeddings is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing")
        dim = get_embedding_dimension()
        kwargs = {
            "model": settings.embedding_model,
            "google_api_key": settings.gemini_api_key,
        }
        if "gemini-embedding" in settings.embedding_model.lower():
            kwargs["output_dimensionality"] = dim
        _embeddings = GoogleGenerativeAIEmbeddings(**kwargs)
    return _embeddings




def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    
    index_list = pc.list_indexes()
    if hasattr(index_list, "names"):
        names = index_list.names()
    else:
        names = [getattr(x, "name", x["name"]) for x in index_list]

    if settings.pinecone_index_name in names:
        index_info = pc.describe_index(settings.pinecone_index_name)
        current_dimension = getattr(index_info, "dimension", None)
        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        if current_dimension is not None and current_dimension != desired_dimension:
            raise ValueError(
                f"Pinecone index '{settings.pinecone_index_name}' exists with dimension {current_dimension}, "
                f"but embedding model '{settings.embedding_model}' requires dimension {desired_dimension}. "
                "To prevent accidental data loss, existing indexes are never automatically deleted. "
                "Please configure a matching PINECONE_INDEX_NAME in .env or create a new index with matching dimension."
            )

    if settings.pinecone_index_name not in names:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        for _ in range(60):
            info = pc.describe_index(settings.pinecone_index_name)
            is_ready = getattr(info.status, "ready", None)
            if is_ready is None and isinstance(info.status, dict):
                is_ready = info.status.get("ready", False)
            if is_ready:
                break
            time.sleep(1)

    return pc.Index(settings.pinecone_index_name)





def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )
    return _vectorstore



def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.top_k})



def add_documents(chunks):
    store = get_vectorstore()
    return store.add_documents(chunks)
