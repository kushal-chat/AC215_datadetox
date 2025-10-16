from agents import (
    Agent,
)
from pydantic import BaseModel
from typing import Sequence
from .prompts.prompt import Prompt
from google.cloud import storage

storage_client = storage.Client()

search_agent = Agent(
    name="SearchAgent",
    instructions=Prompt.get_hf_search_prompt(),
    model="gpt-5-mini-2025-08-07", 
    output_type=str,
)
