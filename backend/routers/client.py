from fastapi import APIRouter
from pydantic import BaseModel
import logging
from rich.logging import RichHandler
from agents import Runner

from .search import search_agent

router = APIRouter(prefix="/flow")

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)


class Query(BaseModel):
    query_val: str


@router.post("/search")
async def run_search(query: Query) -> dict:
    search_logger = logger.getChild("search")

    search_logger.info(f"Query '{query.query_val}' is running.")

    res = await Runner.run(search_agent, input=f"Query: {query.query_val}")

    search_logger.info(f"Query '{query.query_val}' is done running.")

    return {"result": res.final_output_as(str)}
