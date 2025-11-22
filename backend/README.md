# `datadetox_agents.py/rag_cli.py`

## Workflow 

The RAG workflow consists of two main components — a ChromaDB instance hosted on GCP Cloud Run and a GCP bucket with documentations of popular foundation models in markdown format. `rag.py` contains a @function_tool `search_model_doc` which is accessible by an LLM agent. It has the following parameters: 
- `init_db`: Flag to initialise the database (default: False)
- `bucket_name`: Name of the GCP bucket containing model docs (default: "datadetox")
- `prefix`: Folder name for model doc files in the GCP bucket (default: "model_doc")
- `collection_name`: Name of the ChromaDB collection (default: "hf_foundation_models")
- `chunk_size`: Size of text chunks in characters (default: 256)
- `chunk_overlap`: Overlap between chunks in characters (default: 32)
- `query`: Query that can be used to retrieve relevant chunks (default: None) 

P.S. The ChromaDB Cloud Run instance is charged by request so we do not need to worry about it sitting in idle while active. 

## Set Up 

### 1. `secrets/` 

You will need GCP credientials to access the GCP bucket that stores the model doc files. To obtain it, on the Navigation Menu, go to `IAM & Admin > Service Accounts` (or just search up "Service Accounts"). Find `chroma-runner` and click into it. Under the "Keys" tab, generate a new JSON key and save it to `secrets/` folder (the folder is gitignored). 

### 2. `.env` variables 

Copy the variables in `.env.example` onto your `.env` file which is gitignored. 

#### `CHROMA_HOST` 

To find the host address for ChromaDB on GCP, go to GCP Console. On the Navigation Menu, go to "Cloud Run", then "Services" (Alternatively just search for "Cloud Run"). Click into "chromadb" instance, copy the URL and paste it to `.env` for `CHROMA_HOST`. 

#### `PROJECT_ID` 

To find your project id, go to the top left of your GCP console, click on the name of your project, and find the ID. 

#### `GCP_CREDENTIALS_FILE`

The filename of your JSON file for GCP credentials. 