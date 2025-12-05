import json
from agents import RunResultStreaming, Runner
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel

import logging
from rich.logging import RichHandler

from .search import compiler_agent, hf_search_agent, neo4j_search_agent
from .search.utils.tool_state import get_tool_result, set_request_context

router = APIRouter(prefix="/flow")

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)

class Query(BaseModel):
    query_val: str

@router.post("/search")
async def run_search(query: Query, request: Request):
    search_logger = logger.getChild("search")
    request.state.tool_results = {}
    request.state.original_query = query.query_val
    set_request_context(request)

    search_logger.info(f"Query '{query.query_val}' is running.")

    ### HF
    try:
        hf_result: RunResultStreaming = Runner.run_streamed(
            starting_agent=hf_search_agent, 
            input=query.query_val
        )
        async for event in hf_result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield (event.data.delta)

    except Exception as e:
        search_logger.error(f"Failed to initialize Runner: {e}")
        raise 

    ### NEO4J
    ### TODO: TEST WITH RUNNING NEO4J
    try:
        neo4j_result: RunResultStreaming = Runner.run_streamed(
            starting_agent=neo4j_search_agent, 
            input=hf_result.final_output_as(str)
        )
        async for event in neo4j_result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield (event.data.delta)

    except Exception as e:
        search_logger.error(f"Failed to initialize Runner: {e}")
        raise 

    ### TODO: YIELD TREE?

    ### HF ON NEW MODELS
    try:
        hf_result: RunResultStreaming = Runner.run_streamed(
            starting_agent=hf_search_agent, 
            input=query.query_val
        )
        async for event in neo4j_result.final_output_as(str):
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield (event.data.delta)

    except Exception as e:
        search_logger.error(f"Failed to initialize Runner: {e}")
        raise 

    ### FINAL
    try:
        compiled_response: RunResultStreaming = Runner.run_streamed(
            starting_agent=compiler_agent, 
            input=query.query_val
        )
        async for event in compiled_response.final_output_as(str):
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield (event.data.delta)

    except Exception as e:
        search_logger.error(f"Failed to initialize Runner: {e}")
        raise 