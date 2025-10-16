"""
This module provides the FastAPI router for handling client requests.

It manages text-based queries and orchestrates the AI agent workflow.
"""

import fastapi
import asyncio
from fastapi import APIRouter, status
from pydantic import BaseModel
import logging
from logging import StreamHandler
from rich.logging import RichHandler

# TODO To implement later. https://weave-docs.wandb.ai/guides/integrations/openai_agents/
# import weave
# from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor

from agents import (
    Runner,
)

#############   DATADETOX AGENTS (as of MS2)   #############
from datadetox_agents.search_agent import search_agent
# from agents.reason_agent import ReasonAgent...
# etc. 

# Create router
router = APIRouter(prefix="/client", tags=["client"])

logging.basicConfig(level=logging.DEBUG, handlers=[RichHandler()])
logger = logging.getLogger(__name__)

class BasicOutput(BaseModel):
    """
    Health check response
    """
    status: str = "I am healthy!"

@router.post(
    "/search",
    status_code=status.HTTP_200_OK,
)
async def run_search(request: str) -> str:
    """Search model. 
    
    This endpoint enables queries in a text format for HuggingFace models.
    The query comes from the frontend. The flow is frontend > /client/search. Routing occurs in ./main.py.
    Implemented for Milestone 2.

    Can utilize this endpoint to call the search agent and in-depth insights
    into a particular model on HuggingFace.

    Args:
        request, str

    Returns:
        output, str
    """

    search_logger = logger.getChild("search")

    try:
        logger.info(f"Query {request} is running.")
        result = await Runner.run(search_agent, input=request)
        return result.final_output

    except asyncio.exceptions.CancelledError:
        return {"result": "Stopped"}
        
    except Exception as e:
        return {"result": f"Error: {str(e)}"}