from typing import AsyncIterator

from agents import RunResultStreaming, Runner
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel

import logging
from rich.logging import RichHandler

from .search import compiler_agent, hf_search_agent, neo4j_search_agent
from .search.utils.tool_state import set_request_context

router = APIRouter(prefix="/flow")

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)


class Query(BaseModel):
    query_val: str


async def _stream_response_text(result: RunResultStreaming) -> AsyncIterator[str]:
    """Yield text deltas for a streaming agent run."""
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            delta = event.data.delta
            if delta:
                yield delta


@router.post("/search")
async def run_search(query: Query, request: Request):
    search_logger = logger.getChild("search")
    request.state.tool_results = {}
    request.state.original_query = query.query_val
    set_request_context(request)

    search_logger.info(f"Query '{query.query_val}' is running.")

    async def search_pipeline() -> AsyncIterator[str]:
        ### HF
        try:
            hf_result: RunResultStreaming = Runner.run_streamed(
                starting_agent=hf_search_agent,
                input=query.query_val,
            )
            async for chunk in _stream_response_text(hf_result):
                yield chunk
        except Exception as e:
            search_logger.error(f"Failed running HF search agent: {e}")
            raise

        ### NEO4J
        ### TODO: TEST WITH RUNNING NEO4J
        try:
            neo4j_result: RunResultStreaming = Runner.run_streamed(
                starting_agent=neo4j_search_agent,
                input=hf_result.final_output_as(str),
            )
            async for chunk in _stream_response_text(neo4j_result):
                yield chunk
        except Exception as e:
            search_logger.error(f"Failed running Neo4j search agent: {e}")
            raise

        ### TODO: YIELD TREE?

        ### HF ON NEW MODELS
        try:
            hf_followup: RunResultStreaming = Runner.run_streamed(
                starting_agent=hf_search_agent,
                input=neo4j_result.final_output_as(str),
            )
            async for chunk in _stream_response_text(hf_followup):
                yield chunk
        except Exception as e:
            search_logger.error(f"Failed running follow-up HF agent: {e}")
            raise

        ### FINAL
        try:
            compiled_response: RunResultStreaming = Runner.run_streamed(
                starting_agent=compiler_agent,
                input=query.query_val,
            )
            async for chunk in _stream_response_text(compiled_response):
                yield chunk
        except Exception as e:
            search_logger.error(f"Failed running compiler agent: {e}")
            raise

    return StreamingResponse(
        search_pipeline(),
        media_type="text/event-stream",
    )
