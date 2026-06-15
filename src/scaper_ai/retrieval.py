from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pyprojroot import here

from .config import settings

embeddings = OllamaEmbeddings(model=settings.embed_model)

vector_store = Chroma(
    collection_name="caresheets",
    embedding_function=embeddings,
    persist_directory=str(here("db/chroma_db")),
)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.4,
    },
)

chat = ChatOllama(model=settings.chat_model)
