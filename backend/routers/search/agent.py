from typing import List
from agents import Agent, FunctionTool
from .utils import search_huggingface, search_neo4j

tools: List[FunctionTool] = [search_huggingface, search_neo4j]

hf_search_agent = Agent(
    name="HFSearchAgent",
    instructions="Run search_huggingface() to get info from HuggingFace, and get the model_id or dataset_id.",
    model="gpt-5-nano",
    tools=[search_huggingface],
)

neo4j_search_agent = Agent(
    name="Neo4jInfoAgent",
    instructions="Get model ID from the input and search_neo4j(model_id) with the model ID to get info on connected, similar models / datasets.",
    model="gpt-5-nano",
    tools=[search_neo4j],
)

compiler_agent = Agent(
    name="CompilerAgent",
    instructions="Take the information and provide a report of your findings.",
    model="gpt-5.1",
)
