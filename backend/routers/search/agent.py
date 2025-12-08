from typing import List
from agents import Agent, FunctionTool, StopAtTools
from .utils import search_huggingface, search_neo4j
from .utils.extract_datasets import extract_training_datasets

instructions = """
    Receive an input of a model or a dataset.
    First,
    - search_huggingface() to get info from HuggingFace, and get the model_id or dataset_id.
    Second,
    - search_neo4j(model_id) with the model ID to get info on connected, similar models / datasets.
    Third,
    - search_huggingface() to get information on those connected models and datasets from HuggingFace.
    Fourth,
    - extract_training_datasets() to extract training dataset information from arxiv papers for the models found in the neo4j tree search.
      Pass all model IDs from the neo4j results to this tool for efficient parallel processing.
    Finally,
    - Summarize your findings, including the training datasets found in the arxiv papers.
    """

tools: List[FunctionTool] = [
    search_huggingface,
    search_neo4j,
    extract_training_datasets,
]

search_agent = Agent(
    name="SearchAgent",
    instructions=instructions,
    model="gpt-5-nano",
    tools=tools,
    tool_use_behavior=StopAtTools(stop_at_tool_names=["search_neo4j"]),
)
