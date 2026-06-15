"""
Script to create and populate a vector store from documents in a given file path.

Usage:
    uv run scripts/02_populate_vector_store.py <path_to_documents>
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyprojroot import here

from scaper_ai import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_documents(data_path: Path) -> list[Document]:
    """
    Load all .txt files from the given directory.

    Args:
        data_path: Path to directory containing .txt files

    Returns:
        List of Document objects
    """
    documents: list[Document] = []

    if not data_path.exists():
        raise FileNotFoundError(f"Path does not exist: {data_path}")

    if not data_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {data_path}")

    for file in data_path.glob("*.txt"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()

            doc = Document(page_content=content, metadata={"source": str(file)})
            documents.append(doc)

        except UnicodeDecodeError as e:
            logger.warning(f"Error reading {file}: {e}")
            continue

    return documents


def split_documents(
    documents: list[Document], chunk_size: int = 2000, chunk_overlap: int = 500
) -> list[Document]:
    """
    Split documents into smaller chunks.

    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        List of split Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


def populate_vector_store(
    documents: list[Document],
    embedding_model: str = "nomic-embed-text",
    collection_name: str = "caresheets",
    persist_directory: Path | None = None,
    fresh_store: bool = False,
) -> Chroma:
    """
    Create and populate a Chroma vector store with documents.

    Args:
        documents: List of split Document objects
        embedding_model: Name of the embedding model to use
        collection_name: Name of the Chroma collection
        persist_directory: Directory to persist the vector store (defaults to chroma_db)
        fresh_store: If True, remove the existing persist_directory before creating a new store

    Returns:
        Chroma vector store instance
    """
    if persist_directory is None:
        persist_directory = here("db/chroma_db")

    if fresh_store and persist_directory.exists():
        shutil.rmtree(persist_directory)

    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    vector_store.add_documents(documents=documents)
    return vector_store


def main():
    parser = argparse.ArgumentParser(
        description="Create and populate a vector store from documents."
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to directory containing .txt documents",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Size of each document chunk in characters (default: 2000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=500,
        help="Overlap between chunks in characters (default: 500)",
    )
    parser.add_argument(
        "--embedding-model",
        default=settings.embed_model,
        help="Name of the embedding model to use (default: nomic-embed-text)",
    )
    parser.add_argument(
        "--collection-name",
        default="caresheets",
        help="Name of the Chroma collection (default: caresheets)",
    )
    parser.add_argument(
        "--persist-directory",
        type=Path,
        help="Directory to persist the vector store (default: chroma_db)",
    )
    parser.add_argument(
        "--fresh-store",
        action="store_true",
        help="Remove any existing persisted vector store before creating a fresh one",
    )

    args = parser.parse_args()

    try:
        # Load documents
        logger.info(f"Loading documents from {args.data_path}...")
        documents = load_documents(args.data_path)

        if not documents:
            logger.info("No documents found. Exiting.")
            return

        logger.info(f"Loaded {len(documents)} documents.")
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chars = total_chars / len(documents)
        logger.info(f"Total characters: {total_chars:,}")
        logger.info(f"Average characters per document: {avg_chars:.0f}")

        # Split documents
        logger.info(
            f"Splitting documents (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})..."
        )
        split_docs = split_documents(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        logger.info(f"Split into {len(split_docs)} chunks.")

        # Populate vector store
        logger.info(
            f"Populating vector store (collection: {args.collection_name}) using {args.embedding_model!r}..."
        )
        vector_store = populate_vector_store(
            split_docs,
            embedding_model=args.embedding_model,
            collection_name=args.collection_name,
            persist_directory=args.persist_directory,
            fresh_store=args.fresh_store,
        )
        logger.info(f"Successfully added {len(split_docs)} documents to vector store.")
        logger.info(
            f"Vector store persisted to: {vector_store._client.get_settings().persist_directory}"
        )

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except NotADirectoryError as e:
        logger.error(f"Not a directory: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
