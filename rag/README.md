# RAG CLI

A command-line interface for document ingestion and querying using ChromaDB and LlamaIndex for Retrieval-Augmented Generation (RAG) applications.

## Overview

This RAG CLI tool provides functionality to:
- Initialise and populate the vector database from a directory by: 
    - Ingest markdown documents from a directory
    - Chunk documents using sentence splitting
    - Store documents in ChromaDB with vector embeddings
- Query the database for relevant document chunks

## Features

- **Database Initialisation**: Automatically reads all `.md` files from a specified directory, chunk and embed the files into a ChromaDB vector storage 
- **Query Interface**: Simple command-line interface for querying the document database 

## Set up 

To run the containerised pipeline with docker, first start the `rag_cli` and `chromadb` containers by inputting the following commands at the root level of the project: 
```bash
docker compose up -d chromadb
docker compose up -d rag_cli
```

Go inside the rag_cli container with: 
```bash
docker compose exec rag_cli bash 
```

## Usage

### Initialise Database

To create a new database collection from markdown documents:

```bash
uv run python rag_cli.py --init_db=True --input_data_path="data/model_doc" --collection_name="hf_foundation_models"
```

**Parameters:**
- `--init_db`: Flag to initialise the database
- `--input_data_path`: Path to directory containing markdown files (default: "data/model_doc")
- `--collection_name`: Name of the ChromaDB collection (default: "hf_foundation_models")
- `--chunk_size`: Size of text chunks in characters (default: 256)
- `--chunk_overlap`: Overlap between chunks in characters (default: 32)

### Query Database

To query the existing database:

```bash
uv run python rag_cli.py --query=<enter-your-query> --collection_name="hf_foundation_models" --n_results=5
```

**Parameters:**
- `--query`: The search query string
- `--collection_name`: Name of the ChromaDB collection to query
- `--n_results`: Number of results to return (default: 5)
