from agents import Runner
from fastapi import APIRouter, Request
from pydantic import BaseModel

import logging
from rich.logging import RichHandler

from .search import search_agent
from .search.utils.tool_state import get_tool_result, set_request_context
from .search.utils.dataset_resolver import enrich_dataset_info

router = APIRouter(prefix="/flow")

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)


class Query(BaseModel):
    query_val: str


@router.post("/search")
async def run_search(query: Query, request: Request) -> dict:
    search_logger = logger.getChild("search")

    # Initialize tool results storage in request state
    request.state.tool_results = {}

    # Store the original user query for later use
    request.state.original_query = query.query_val

    # Store request in context so tool functions can access it
    set_request_context(request)

    search_logger.info(f"Query '{query.query_val}' is running.")

    res = await Runner.run(search_agent, input=f"Query: {query.query_val}")

    search_logger.info(f"Query '{query.query_val}' is done running.")

    # Get the stored neo4j result from request state
    neo4j_result = get_tool_result("search_neo4j", request)
    datasets_result = get_tool_result("extract_training_datasets", request)

    response = {"result": res.final_output_as(str)}

    # Add neo4j_data if available
    if neo4j_result is not None:
        neo4j_data = neo4j_result.model_dump()

        # Merge dataset information into the nodes
        if datasets_result is not None and isinstance(datasets_result, dict):
            # Create a mapping of model_id to dataset info
            for node in neo4j_data.get("nodes", {}).get("nodes", []):
                model_id = node.get("model_id")
                if model_id and model_id in datasets_result:
                    dataset_info = datasets_result[model_id]
                    # Add training_datasets field to the node
                    node["training_datasets"] = {
                        "arxiv_url": dataset_info.get("arxiv_url"),
                        "datasets": [],
                    }

                    # Enrich dataset information with HuggingFace URLs
                    raw_datasets = dataset_info.get("datasets", [])
                    enriched_datasets = enrich_dataset_info(raw_datasets)
                    node["training_datasets"]["datasets"] = enriched_datasets

        response["neo4j_data"] = neo4j_data

    return response
