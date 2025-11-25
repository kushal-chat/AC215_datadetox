from typing import List
from agents import Agent, FunctionTool, StopAtTools
from .utils import search_huggingface, search_neo4j 

instructions = """
    Receive an input of a model or a dataset.
    First, 
    - search_huggingface() to get info from HuggingFace, and get the model_id.
    Second,
    - search_neo4j(model_id) with the model ID to get info on connected, similar models / datasets.
    Third,
    - search_huggingface() to get information on those connected models and datasets from HuggingFace.
    Finally, 
    - Summarize your findings.
    """

tools: List[FunctionTool] = [search_huggingface, search_neo4j]

search_agent = Agent(
    name="SearchAgent",
    instructions=instructions,
    model="gpt-5-nano",
    tools=tools,
    tool_use_behavior=StopAtTools(stop_at_tool_names=["search_neo4j"]),
)
