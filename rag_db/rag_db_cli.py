import os 
import fire 
import logging
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core.text_splitter import SentenceSplitter

import chromadb
from chromadb.types import Collection
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN", None)

def ingest_md_data(input_data_path: str): 
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

def init_db_client(port: int=5432): 
    chroma_client = chromadb.HttpClient(host="chroma", port = port)
    # chroma_client = chromadb.PersistentClient(path="data/chroma")
    logger.info("Initialised ChromaDB client")
    return chroma_client 

def create_db_collection(chroma_client, collection_name: str): 
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

def main(input_data_path: str, collection_name: str, chunk_size: int=256, chunk_overlap: int=32): 
    md_files = ingest_md_data(input_data_path)
    chunked_files = chunk_md_data(md_files, chunk_size, chunk_overlap)
    chroma_client = init_db_client()
    collection = create_db_collection(chroma_client, collection_name)
    populate_db_collection(collection, chunked_files)

if __name__ == "__main__":
    fire.Fire(main)