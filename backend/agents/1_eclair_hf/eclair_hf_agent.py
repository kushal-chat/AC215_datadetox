from agents import (
    Agent,
    WebSearchTool, 
    handoff,
)
from .guardrail_agent import sensitive_guardrail
from pydantic import BaseModel
from typing import Sequence
from context import ShinanContext, context_tool
from ..prompts import Prompt
from agents.model_settings import ModelSettings
from agents.extensions import handoff_filters

class TextSearchIdea(BaseModel):
    """
    A search idea. 
    Defines:
    - query: The search idea.
    - reasoning: Why this is a good search idea in context of the text.
    """
    query: str
    reasoning: str

class TextSearchIdeas(BaseModel):
    """A list of search ideas."""
    ideas: Sequence[TextSearchIdea]

SEARCH_PROMPT = Prompt().get_text_prompt()

text_agent = Agent[ShinanContext](
    name="TextAgent",
    instructions=SEARCH_PROMPT,
    model="gpt-4.1-nano-2025-04-14", 
    model_settings=ModelSettings(tool_choice="required"),
    output_type=TextSearchIdeas,
    tools=[context_tool],
    input_guardrails=[sensitive_guardrail],
)
