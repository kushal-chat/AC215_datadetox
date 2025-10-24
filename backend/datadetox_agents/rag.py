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
import json
import fire
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from google.cloud import storage
from llama_index.core.text_splitter import SentenceSplitter
from pydantic import BaseModel
from typing import List, Optional

import chromadb
from chromadb.types import Collection
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from agents import function_tool

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get environment variables
HF_TOKEN = os.getenv("HF_TOKEN", None)
PROJECT_ID = os.getenv("PROJECT_ID", None)
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", 8000)
GCP_CREDENTIALS_FILE = os.getenv("GCP_CREDENTIALS_FILE", None)

class QueryResultItem(BaseModel):
    id: str
    document: str
    distance: float
    metadata: Optional[dict] = None

class QueryResponse(BaseModel):
    query: str
    collection_name: str
    results: List[QueryResultItem]


### Initialise Clients ### 
def init_db_client(host: str=CHROMA_HOST, port: int=CHROMA_PORT): 
    """
    initialise ChromaDB client connection.

    Args:
        port (int): Port number for ChromaDB HTTP server

    Returns:
        ChromaClient: initialised ChromaDB client
    """
    chroma_client = chromadb.HttpClient(host=host, port=port)
    logger.info("Initialised ChromaDB client")
    return chroma_client 

def init_gcp_client(): 
    """
    initialise GCP client connection.

    Returns:
        GCPClient: initialised GCP client
    """
    gcp_client = storage.Client.from_service_account_json(os.path.join(os.path.dirname(__file__), "..", "..", "secrets", GCP_CREDENTIALS_FILE))
    logger.info("Initialised GCP client")
    return gcp_client 

### Ingest Data ### 

def ingest_md_data(bucket_name: str, prefix: str): 
    """
    Ingest markdown files from a GCS bucket.

    Args:
        bucket_name (str): Name of the GCS bucket
        prefix (str): Prefix of the files to ingest
    """
    gcp_client = init_gcp_client()
    try:
        bucket = gcp_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
    except Exception as e:
        logger.error(f"Failed to access bucket {bucket_name} with prefix {prefix}: {e}")
        raise

    md_files = []
    for blob in blobs:
        if blob.name.endswith(".md"):
            try: 
                md_files.append({
                    "model_name": blob.name,
                    "content": blob.download_as_text(encoding="utf-8")
                })
                logger.info(f"Read file: {blob.name}")
            except Exception as e:
                logger.error(f"Error reading file {blob.name}: {e}")
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
        model_name = md_file["model_name"].split("/")[-1].replace(".md", "")
        content = md_file["content"]
        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            chunked_files["chunk_id"].append(f"{model_name}_{i}")
            chunked_files["chunk"].append(chunk)
            chunked_files["metadata"].append({
                "model_name": model_name,
            })
    return chunked_files

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
        logger.info(f"Collection '{collection_name}' does not exist. Creating new.")
    
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

def init_database(bucket_name: str, prefix: str, collection_name: str, chunk_size: int=256, chunk_overlap: int=32):
    """
    initialise the database with documents.

    Args:
        input_data_path (str): Path to directory containing markdown files
        collection_name (str): Name for the ChromaDB collection
        chunk_size (int): Size of text chunks in characters
        chunk_overlap (int): Overlap between chunks in characters
    """
    md_files = ingest_md_data(bucket_name, prefix)
    chunked_files = chunk_md_data(md_files, chunk_size, chunk_overlap)
    chroma_client = init_db_client()
    collection = create_db_collection(chroma_client, collection_name)
    populate_db_collection(collection, chunked_files)
    logger.info("Database initialisation complete")
    logger.info(f"Collection '{collection_name}' created with {collection.count()} entries")

def query_rag(query: str, collection_name: str, n_results: int = 5) -> QueryResponse:
    """Query the existing database and return structured results."""
    chroma_client = init_db_client()
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=DefaultEmbeddingFunction()
        )
        logger.info(f"Collection '{collection_name}' found")
    except Exception:
        logger.error(f"Collection '{collection_name}' not found. Please initialise the database first.")
        return QueryResponse(query=query, collection_name=collection_name, results=[])

    results = collection.query(
        query_texts=query,
        n_results=n_results,
    )

    # Convert raw results into Pydantic structure
    structured_results = []
    for idx, (chunk_id, doc, dist, meta) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        )
    ):
        structured_results.append(
            QueryResultItem(
                id=chunk_id,
                document=doc,
                distance=dist,
                metadata=meta,
            )
        )

    response = QueryResponse(
        query=query,
        collection_name=collection_name,
        results=structured_results,
    )
    
    logger.info(f"Query returned {len(structured_results)} results from '{collection_name}'")
    return response

@function_tool
def search_model_doc(
    init_db: bool = False,
    bucket_name: str = "datadetox",
    prefix: str = "model_doc",
    collection_name: str = "hf_foundation_models",
    query: str = "deepseek",
    n_results: int = 5,
    chunk_size: int = 256,
    chunk_overlap: int = 32,
) -> Optional[QueryResponse]:
    """Search the database for relevant model documents."""
    if init_db:
        if not bucket_name or not prefix:
            logger.error("bucket_name and prefix are required for database initialisation")
            return None
        try:
            init_database(bucket_name, prefix, collection_name, chunk_size, chunk_overlap)
        except Exception as e:
            logger.error(f"Error accessing GCP bucket: {str(e)}")
            return None
        return None

    elif query:
        return query_rag(query, collection_name, n_results)

    else:
        logger.error("Please specify either --init_db or --query")
        return None

if __name__ == "__main__":
    fire.Fire(search_model_doc)