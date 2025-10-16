"""
RAG CLI tool for document ingestion and querying using ChromaDB.

This module provides functionality to:
- initialise and populate a vector database from markdown documents
- Chunk and embed documents using sentence splitting
- Query the database for relevant document chunks

The CLI supports two main operations:
1. Database initialisation (--init_db)
2. Document querying (--query)
"""

import os 
import fire 
import logging
from rich.logging import RichHandler

from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core.text_splitter import SentenceSplitter

import chromadb
from chromadb.types import Collection
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)

# Get Hugging Face token from environment
HF_TOKEN = os.getenv("HF_TOKEN", None)

def ingest_md_data(input_data_path: str): 
    """
    Read and parse markdown files from the specified directory.

    Args:
        input_data_path (str): Path to directory containing markdown files

    Returns:
        list[dict]: List of dictionaries containing model names and file contents
        
    Raises:
        FileNotFoundError: If input path does not exist
        NotADirectoryError: If input path is not a directory
    """
    logger.info(f"Reading files from {input_data_path}")
    input_path = Path(input_data_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_data_path} does not exist")

    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path {input_data_path} is not a directory")

    md_files = []
    for file in input_path.glob("**/*.md"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                model_name = str(file).split("/")[-1].split(".")[0]
                md_files.append({
                    "model_name": model_name,
                    "content": content
                })
            logger.info(f"Read file: {model_name}")
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")
            continue

    logger.info(f"Found {len(md_files)} files")
    return md_files
    
def chunk_md_data(md_files: list[dict], chunk_size: int=256, chunk_overlap: int=32): 
    """
    Split markdown documents into overlapping chunks for embedding.

    Args:
        md_files (list[dict]): List of dictionaries containing model names and contents
        chunk_size (int): Size of text chunks in characters
        chunk_overlap (int): Overlap between chunks in characters

    Returns:
        dict: Dictionary containing chunk IDs, chunks and metadata
    """
    text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_files = {"chunk_id": [], "chunk": [], "metadata": []}
    for md_file in md_files:
        model_name = md_file["model_name"]
        content = md_file["content"]
        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            chunked_files["chunk_id"].append(f"{model_name}_{i}")
            chunked_files["chunk"].append(chunk)
            chunked_files["metadata"].append({
                "model_name": model_name,
            })
    return chunked_files

def init_db_client(port: int=8000): 
    """
    initialise ChromaDB client connection.

    Args:
        port (int): Port number for ChromaDB HTTP server

    Returns:
        ChromaClient: initialised ChromaDB client
    """
    chroma_client = chromadb.HttpClient(host="chromadb", port = port)
    logger.info("Initialised ChromaDB client")
    return chroma_client 

def create_db_collection(chroma_client, collection_name: str): 
    """
    Create a new ChromaDB collection, deleting existing one if it exists.

    Args:
        chroma_client: ChromaDB client instance
        collection_name (str): Name for the collection

    Returns:
        Collection: Created ChromaDB collection
    """
    try:
        # Clear out any existing items in the collection
        chroma_client.delete_collection(name=collection_name)
        logger.info(f"Deleted existing collection '{collection_name}'")
    except Exception:
        logger.info(f"Collection '{collection_name}' did not exist. Creating new.")
    
    collection = chroma_client.create_collection(
        name=collection_name, 
        metadata={"hnsw:space": "cosine"}, 
        embedding_function=DefaultEmbeddingFunction()
        )
    logger.info(f"Created collection: {collection_name}")
    return collection 

def populate_db_collection(collection: Collection, chunked_files: dict[list]): 
    """
    Populate ChromaDB collection with chunked documents.

    Args:
        collection (Collection): ChromaDB collection to populate
        chunked_files (dict[list]): Dictionary containing chunks and metadata
    """
    total_chunks = len(chunked_files["chunk_id"])
    batch_size = 100  # Process in batches to show progress
    
    with tqdm(total=total_chunks, desc="Populating collection") as pbar:
        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            collection.add(
                ids=chunked_files["chunk_id"][i:batch_end],
                documents=chunked_files["chunk"][i:batch_end], 
                metadatas=chunked_files["metadata"][i:batch_end]
            )
            pbar.update(batch_end - i)
    logger.info(f"Populated collection: {collection.name}") 

def init_database(input_data_path: str, collection_name: str, chunk_size: int=256, chunk_overlap: int=32):
    """
    initialise the database with documents.

    Args:
        input_data_path (str): Path to directory containing markdown files
        collection_name (str): Name for the ChromaDB collection
        chunk_size (int): Size of text chunks in characters
        chunk_overlap (int): Overlap between chunks in characters
    """
    md_files = ingest_md_data(input_data_path)
    chunked_files = chunk_md_data(md_files, chunk_size, chunk_overlap)
    chroma_client = init_db_client()
    collection = create_db_collection(chroma_client, collection_name)
    populate_db_collection(collection, chunked_files)
    logger.info("Database initialisation complete")
    logger.info(f"Collection '{collection_name}' created with {collection.count()} entries")

def query_rag(query: str, collection_name: str, n_results: int=5):
    """
    Query the existing database.

    Args:
        query (str): Query string to search for
        collection_name (str): Name of ChromaDB collection to query
        n_results (int): Number of results to return

    Returns:
        dict: Query results from ChromaDB
    """
    chroma_client = init_db_client()
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=DefaultEmbeddingFunction()
        )
        logger.info(f"Collection '{collection_name}' found")
    except Exception:
        logger.error(f"Collection '{collection_name}' not found. Please initialise the database first.")
        return

    results = collection.query(
        query_texts=query,
        n_results=n_results,
    )
    logger.info(f"Query results: {results}")
    return results

def main(
    init_db: bool = False,
    input_data_path: str = "data/model_doc",
    collection_name: str = "hf_foundation_models",
    query: str = None,
    n_results: int = 5,
    chunk_size: int = 256,
    chunk_overlap: int = 32, 
):
    """
    Main CLI function that handles both initialisation and querying.

    Args:
        init_db (bool): Flag to initialise database
        input_data_path (str): Path to directory containing markdown files
        collection_name (str): Name for the ChromaDB collection
        query (str): Query string for searching the database
        n_results (int): Number of results to return for queries
        chunk_size (int): Size of text chunks in characters
        chunk_overlap (int): Overlap between chunks in characters
    """
    if init_db:
        if not input_data_path:
            logger.error("input_data_path is required for database initialisation")
            return
        init_database(input_data_path, collection_name, chunk_size, chunk_overlap)
    elif query:
        query_rag(query, collection_name, n_results)
    else:
        logger.error("Please specify either --init_db or --query")

if __name__ == "__main__":
    fire.Fire(main)