from typing import List
from agents import Agent, FunctionTool, StopAtTools
from .utils import search_huggingface, search_neo4j, search_models, search_datasets

instructions = (
    """
    Receive an input of a model or a dataset.
    First, 
    - search_huggingface() to get info from HuggingFace, and get the model_id.
    Second,
    - search_models() to ensure that the model is in the database.
    Third,
    - search_neo4j(model_id) with the model ID to get info on connected, similar models / datasets.
    Summarize your findings.
    """
)

tools: List[FunctionTool] = [search_huggingface, search_neo4j, search_models]

search_agent = Agent(
    name="SearchAgent",
    instructions=instructions,
    model="gpt-5-nano",
    tools=tools,
    tool_use_behavior=StopAtTools(stop_at_tool_names=["search_neo4j"])
)