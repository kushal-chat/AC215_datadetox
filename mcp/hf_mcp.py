import os
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP
from huggingface_hub import HfApi, ModelInfo, DatasetInfo, SpaceInfo, ModelCard, DatasetCard

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

HOST = os.getenv("HF_MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("HF_MCP_PORT", "8081"))
HF_TOKEN = os.getenv("HF_TOKEN", None)

api = HfApi(
    endpoint="https://huggingface.co",
    token=HF_TOKEN
    )

def create_server():
    mcp = FastMCP(
        name="HuggingFace MCP",
        instructions="""
This server provides tools to search Hugging Face Hub and retrieve repo cards.
Tools:
- hf_search(query, kind, limit, filter)
- hf_card(repo_id, kind)
"""
    )

    @mcp.tool()
    async def hf_search(
        query: str,
        kind: str = "models",  # "models" | "datasets" | "spaces"
        limit: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        logger.info(f"hf_search called with query='{query}', kind='{kind}', limit={limit}")
        
        if not query or not query.strip():
            logger.warning("Empty query received")
            return {"results": []}
        
        filter = filter or {}
        results: List[Dict[str, Any]] = []

        try:
            if kind == "models":
                logger.info(f"Searching for models with query: {query}")
                models = list(api.list_models(search=query, limit=limit, **filter))
                logger.info(f"Found {len(models)} models")
                
                for m in models:
                    assert isinstance(m, ModelInfo)
                    result = {
                        "repo_id": m.modelId,
                        "author": m.author,
                        "sha": m.sha,
                        "likes": m.likes,
                        "downloads": m.downloads,
                        "tags": m.tags,
                        "private": m.private,
                        "library_name": getattr(m, "library_name", None),
                        "url": f"https://huggingface.co/{m.modelId}",
                        "kind": "model",
                    }
                    results.append(result)
                    logger.info(f"Added model: {m.modelId}")
                    
            elif kind == "datasets":
                logger.info(f"Searching for datasets with query: {query}")
                datasets = list(api.list_datasets(search=query, limit=limit, **filter))
                logger.info(f"Found {len(datasets)} datasets")
                
                for d in datasets:
                    assert isinstance(d, DatasetInfo)
                    result = {
                        "repo_id": d.id,
                        "author": d.author,
                        "sha": d.sha,
                        "likes": d.likes,
                        "downloads": d.downloads,
                        "tags": d.tags,
                        "private": d.private,
                        "url": f"https://huggingface.co/datasets/{d.id}",
                        "kind": "dataset",
                    }
                    results.append(result)
                    logger.info(f"Added dataset: {d.id}")
                    
            elif kind == "spaces":
                logger.info(f"Searching for spaces with query: {query}")
                spaces = list(api.list_spaces(search=query, limit=limit, **filter))
                logger.info(f"Found {len(spaces)} spaces")
                
                for s in spaces:
                    assert isinstance(s, SpaceInfo)
                    result = {
                        "repo_id": s.id,
                        "author": s.author,
                        "sha": s.sha,
                        "likes": s.likes,
                        "tags": s.tags,
                        "private": s.private,
                        "runtime": s.runtime,
                        "sdk": s.sdk,
                        "url": f"https://huggingface.co/spaces/{s.id}",
                        "kind": "space",
                    }
                    results.append(result)
                    logger.info(f"Added space: {s.id}")
            else:
                raise ValueError("kind must be one of: models, datasets, spaces")

            logger.info(f"Returning {len(results)} results")
            return {"results": results}
            
        except Exception as e:
            logger.error(f"Error in hf_search: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"results": [], "error": str(e)}

    @mcp.tool()
    async def hf_card(repo_id: str, kind: str = "model") -> Dict[str, Any]:
        logger.info(f"hf_card called with repo_id='{repo_id}', kind='{kind}'")
        
        if not repo_id:
            raise ValueError("repo_id required")
        kind = kind.lower()
        
        try:
            if kind == "model":
                card = api.model_info(repo_id)
                readme = None
                try:
                    readme_card = ModelCard.load(repo_id)
                    readme = readme_card.text if readme_card else None
                except Exception as e:
                    logger.warning(f"Could not fetch model card for {repo_id}: {e}")
                url = f"https://huggingface.co/{repo_id}"
            elif kind == "dataset":
                card = api.dataset_info(repo_id)
                readme = None
                try:
                    readme_card = DatasetCard.load(repo_id)
                    readme = readme_card.text if readme_card else None
                except Exception as e:
                    logger.warning(f"Could not fetch dataset card for {repo_id}: {e}")
                url = f"https://huggingface.co/datasets/{repo_id}"
            elif kind == "space":
                card = api.space_info(repo_id)
                readme = None
                url = f"https://huggingface.co/spaces/{repo_id}"
            else:
                raise ValueError("kind must be model|dataset|space")

            result = {
                "repo_id": repo_id,
                "kind": kind,
                "url": url,
                "card": card.__dict__ if hasattr(card, "__dict__") else str(card),
                "readme": readme,
            }
            logger.info(f"Successfully fetched card for {repo_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching card for {repo_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    return mcp

def main():
    server = create_server()
    logger.info(f"Starting HF MCP on {HOST}:{PORT} (SSE)")
    logger.info(f"HF Token configured: {bool(HF_TOKEN)}")
    server.run(transport="sse", host=HOST, port=PORT)

if __name__ == "__main__":
    main()