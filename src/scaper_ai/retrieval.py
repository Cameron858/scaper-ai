from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pyprojroot import here

from .config import settings


def create_embeddings(model_name: str | None = None) -> OllamaEmbeddings:
    """Create an Ollama embeddings client."""
    if model_name is None:
        model_name = settings.embed_model
    return OllamaEmbeddings(model=model_name)


def create_vector_store(
    embedding_function: OllamaEmbeddings | None = None,
    collection_name: str = "caresheets",
    persist_directory: Path | None = None,
) -> Chroma:
    """Create a Chroma vector store instance."""
    if embedding_function is None:
        embedding_function = create_embeddings()
    if persist_directory is None:
        persist_directory = here("db/chroma_db")

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=str(persist_directory),
    )


def create_retriever(
    vector_store: Chroma | None = None,
    search_type: str = "similarity_score_threshold",
    search_kwargs: dict | None = None,
):
    """Create a retriever from a Chroma vector store."""
    if vector_store is None:
        vector_store = create_vector_store()
    if search_kwargs is None:
        search_kwargs = {
            "k": 5,
            "score_threshold": 0.4,
        }
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def create_chat(model_name: str | None = None) -> ChatOllama:
    """Create a ChatOllama client."""
    if model_name is None:
        model_name = settings.chat_model
    return ChatOllama(model=model_name)
