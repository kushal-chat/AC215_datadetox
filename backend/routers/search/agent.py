from typing import List
from agents import Agent, FunctionTool
from .utils import search_huggingface, search_neo4j
from .utils.extract_datasets import extract_training_datasets

tools: List[FunctionTool] = [
    search_huggingface,
    search_neo4j,
    extract_training_datasets,
]

hf_search_agent = Agent(
    name="HFSearchAgent",
    instructions="Run search_huggingface() to get info from HuggingFace, and get the model_id or dataset_id.",
    model="gpt-5-nano",
    tools=[search_huggingface],
)

neo4j_search_agent = Agent(
    name="Neo4jInfoAgent",
    instructions=(
        "You receive HuggingFace search results as input. Your task is to:\n"
        "1. Extract the FIRST model ID from the input text (look for lines like '**1. [model_id]' or model names like 'Qwen/Qwen3-4B')\n"
        "2. IMMEDIATELY call search_neo4j(model_id) with the extracted model ID\n"
        "3. Do NOT ask the user which model to use - automatically select the first/best match\n"
        "4. If multiple models are listed, use the first one in the list\n"
        "Model IDs typically look like 'author/model-name' (e.g., 'Qwen/Qwen3-4B')\n"
        "Call the tool immediately without asking for confirmation."
    ),
    model="gpt-5-nano",
    tools=[search_neo4j],
)

dataset_extractor_agent = Agent(
    name="ArxivDatasetExtractorAgent",
    instructions=(
        "You receive JSON input that contains a list of HuggingFace model_ids gathered from earlier tools. "
        "Call extract_training_datasets(model_ids=[...]) using ONLY the provided IDs to gather arxiv paper links "
        "and their training datasets. Summarize which datasets were found for each model. "
        "If no valid model IDs are supplied, explain why extraction could not run."
    ),
    model="gpt-5-nano",
    tools=[extract_training_datasets],
)

dataset_risk_agent = Agent(
    name="DatasetRiskAgent",
    instructions=(
        "You are given JSON describing models, their arxiv links, and training datasets with risk flags. "
        "Produce a crisp risk briefing: note synthetic datasets (high risk), English-centric data (geographic bias), and unknown sources. "
        "Call out which models lack dataset info. Provide actionable warnings rather than restating the entire graph."
    ),
    model="gpt-4o-mini",
)

compiler_agent = Agent(
    name="CompilerAgent",
    instructions=(
        "Summarize findings in two sections: 'Model Findings' (queried model, key data, dataset coverage) and 'Dependency Watchlist' (only notable upstream risks or gaps)."
        "Keep the summary concise and focused, make sure to highlight key points, link to huggingface models."
    ),
    model="gpt-5.1",
)
