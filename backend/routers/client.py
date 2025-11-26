from agents import Runner
from fastapi import APIRouter, Request
from pydantic import BaseModel

import logging
from rich.logging import RichHandler

from .search import search_agent
from .search.utils.tool_state import get_tool_result, set_request_context

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

    response = {"result": res.final_output_as(str)}

    # Add neo4j_data if available
    if neo4j_result is not None:
        response["neo4j_data"] = neo4j_result.model_dump()

    return response
