from agents import (
    Agent,
)
from agents.mcp import MCPServerStdio, MCPServerStdioParams
from pydantic import BaseModel
from typing import Sequence
from .prompts.prompt import Prompt
from google.cloud import storage

storage_client = storage.Client()

# Stage 1: Convert user query to HuggingFace search terms
search_term_agent = Agent(
    name="SearchTermAgent",
    instructions=Prompt.get_hf_search_prompt(),
    model="gpt-4o-mini",  # Using mini for simple keyword extraction
    output_type=str,
)

# Stage 2: Use search terms to retrieve info from HuggingFace
# Returns basic info + arXiv paper ID if available
hf_info_agent = Agent(
    name="HFInfoAgent",
    instructions=Prompt.get_hf_info_retrieval_prompt_without_mcp(),
    model="gpt-4o",  # Using GPT-4o for HuggingFace queries
    output_type=str,
)

# Stage 3: Use arXiv MCP to get detailed training data from papers
arxiv_mcp_server = MCPServerStdio(
    name="arxiv",
    params=MCPServerStdioParams(
        command="arxiv-mcp-server",
        args=["--storage-path", "/tmp/papers"],
    )
)

paper_analysis_agent = Agent(
    name="PaperAnalysisAgent",
    instructions=Prompt.get_paper_analysis_prompt(),
    model="gpt-4o",  # Using GPT-4o with arXiv MCP
    output_type=str,
    mcp_servers=[arxiv_mcp_server]
)

# Keep backward compatibility - default to search term agent
search_agent = search_term_agent
