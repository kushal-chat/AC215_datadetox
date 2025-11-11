"""
This module provides the FastAPI router for handling client requests.

It manages text-based queries and orchestrates the AI agent workflow.
"""

import fastapi
import asyncio
import time
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
from datadetox_agents.search_agent import search_term_agent, hf_info_agent, paper_analysis_agent
from datadetox_agents.arxiv_helper import fetch_arxiv_paper, extract_training_data_info
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

class SearchRequest(BaseModel):
    """Request model for search endpoint"""
    query: str

@router.post(
    "/search",
    status_code=status.HTTP_200_OK,
)
async def run_search(request: SearchRequest) -> dict:
    """Search model with HuggingFace + arXiv paper analysis.

    This endpoint uses a 3-stage pipeline:
    1. Convert user query to HuggingFace search terms
    2. Get model/dataset info from HF (including arXiv ID if available)
    3. If arXiv paper found, analyze it for detailed training data info

    The query comes from the frontend. The flow is frontend > /client/search.
    Routing occurs in ./main.py. Implemented for Milestone 2+.

    Args:
        request: SearchRequest with query string

    Returns:
        dict with response from HuggingFace + paper analysis
    """

    search_logger = logger.getChild("search")

    try:
        logger.info(f"Query '{request.query}' is running.")

        # Stage 1: Convert query to search terms
        logger.info("Stage 1: Converting query to HF search terms...")
        stage1_start = time.time()
        search_terms_result = await Runner.run(search_term_agent, input=request.query)
        search_terms = search_terms_result.final_output
        stage1_time = round(time.time() - stage1_start, 2)
        logger.info(f"Search terms: {search_terms} (took {stage1_time}s)")

        # Stage 2: Get detailed info from HF
        logger.info("Stage 2: Retrieving info from HuggingFace Hub...")
        stage2_start = time.time()
        hf_result = await Runner.run(hf_info_agent, input=search_terms)
        hf_response = hf_result.final_output
        stage2_time = round(time.time() - stage2_start, 2)
        logger.info(f"Successfully retrieved HuggingFace information (took {stage2_time}s)")

        # Check if there's an arXiv ID in the response
        arxiv_id = None
        if "ARXIV_ID:" in hf_response:
            parts = hf_response.split("ARXIV_ID:")
            if len(parts) > 1:
                arxiv_id = parts[1].strip().split()[0]  # Get first word after ARXIV_ID:
                hf_response = parts[0].strip()  # Remove the ARXIV_ID line from response
                logger.info(f"Found arXiv ID: {arxiv_id}")

        final_response = hf_response
        stage3_time = None

        # Stage 3: If arXiv paper found, analyze it for training data details
        if arxiv_id:
            logger.info(f"Stage 3: Fetching arXiv paper {arxiv_id} for training data...")
            stage3_start = time.time()
            try:
                # Fetch paper using direct arXiv API
                paper_data = fetch_arxiv_paper(arxiv_id)

                if paper_data:
                    # Extract training data info from paper
                    paper_info = extract_training_data_info(paper_data)
                    stage3_time = round(time.time() - stage3_start, 2)
                    logger.info(f"Successfully fetched paper (took {stage3_time}s)")

                    # Combine HF info with paper analysis
                    final_response = f"{hf_response}\n\n---\n\n## 📚 Research Paper Information\n\n{paper_info}"
                else:
                    stage3_time = round(time.time() - stage3_start, 2)
                    logger.warning(f"Could not fetch paper {arxiv_id}")
                    final_response = f"{hf_response}\n\n---\n\n## ⚠️ Could Not Fetch Paper\n\nCould not retrieve arXiv paper {arxiv_id}. Please visit https://arxiv.org/abs/{arxiv_id} directly."

            except Exception as paper_error:
                logger.error(f"Error fetching paper: {str(paper_error)}", exc_info=True)
                stage3_time = round(time.time() - stage3_start, 2)
                # Add error message to response
                final_response = f"{hf_response}\n\n---\n\n## ⚠️ Paper Fetch Failed\n\nError fetching arXiv paper {arxiv_id}: {str(paper_error)}\n\nPlease visit https://arxiv.org/abs/{arxiv_id} directly."
        else:
            logger.info("No arXiv ID found, skipping paper analysis")

        return {
            "response": final_response,
            "search_terms": search_terms,
            "arxiv_id": arxiv_id,
            "stage1_time": stage1_time,
            "stage2_time": stage2_time,
            "stage3_time": stage3_time,
            "status": "success"
        }

    except asyncio.exceptions.CancelledError:
        logger.warning("Request was cancelled.")
        return {"response": "Request was stopped", "status": "cancelled"}

    except Exception as e:
        logger.error(f"Error processing search: {str(e)}", exc_info=True)
        return {"response": f"Error: {str(e)}", "status": "error"}