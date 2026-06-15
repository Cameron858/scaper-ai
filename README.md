# scraper-ai

A small Python project for collecting fishkeeping caresheet content and turning it into a local vector store for retrieval-based AI workflows.

## What this project does

- Scrapes fishkeeping caresheet pages from fishkeeping.co.uk
- Saves the raw extracted text into the local data/rip directory
- Builds a Chroma vector database from those documents for semantic search and retrieval

## Current workflow

1. Run the scraper to collect caresheet text
2. Feed the saved text files into the vector-store script
3. Use the resulting local embeddings database in db/chroma_db

## Project structure

- scripts/01_scrape_caresheets.py - fetches caresheet pages and writes text files
- scripts/02_populate_vector_store.py - loads text files, chunks them, and populates Chroma
- data/rip/ - scraped output files grouped by timestamp
- db/chroma_db/ - persisted Chroma vector store

## Quick start

### 1. Install dependencies

This project uses the Python environment managed by the repository.

```sh
uv sync
```

### 2. Scrape caresheet data

```sh
python scripts/01_scrape_caresheets.py
```

This creates a timestamped folder under data/rip/ with one text file per caresheet.

### 3. Populate the vector store

After scraping, point the vector-store script at the generated folder:

```sh
python scripts/02_populate_vector_store.py data/rip/20260612_113827
```

You can override the embedding model or collection name if needed:

```sh
python scripts/02_populate_vector_store.py data/rip/20260612_113827 \
  --embedding-model nomic-embed-text \
  --collection-name caresheets
```

## Notes

- The vector-store script expects an Ollama-compatible embedding model to be available.
- The default embedding model is `nomic-embed-text`.