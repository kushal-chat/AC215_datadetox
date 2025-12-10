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
        "1. Extract the FIRST model ID or dataset ID from the input text (look for lines like '**1. [model_id]' or model/dataset names like 'Qwen/Qwen3-4B' or 'allenai/c4')\n"
        "2. IMMEDIATELY call search_neo4j(model_id) with the extracted ID (works for both models and datasets)\n"
        "3. Do NOT ask the user which model/dataset to use - automatically select the first/best match\n"
        "4. If multiple models/datasets are listed, use the first one in the list\n"
        "IDs typically look like 'author/model-name' or 'author/dataset-name' (e.g., 'Qwen/Qwen3-4B' or 'allenai/c4')\n"
        "Call the tool immediately without asking for confirmation."
    ),
    model="gpt-4.1-mini",
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
        """Produce a comprehensive, well-organized report.
    Use the section structure below, but you may format naturally as long as the organization is clear.
    Target ~8–12 sentences per section as a guideline, but feel free to expand when useful.

    Model Findings
    - Summarize all findings for the queried models and datasets.
    - Use clear bullets or short paragraphs; sub-bullets are fine if they improve clarity.
    - Include model purpose, scale, architecture notes, dataset provenance, and information extracted from tools.
    - Provide Hugging Face links for any referenced model_ids or dataset_ids.
    - Acknowledge incomplete or missing tool data and interpret what that uncertainty means.

    Dependency Watchlist
    - Highlight upstream risks: synthetic data amplification, unclear or missing provenance, weak or conflicting licenses, dataset bias, auto-generated content contamination, etc.
    - Briefly explain why each risk matters and give a short, actionable recommendation (e.g., sample audit, dataset swap, license chain verification).
    - Call out models or datasets with especially sparse documentation or ambiguous training sources.

    Dataset-focused queries:
    - If the user is primarily asking about a dataset, tailor the findings toward dataset quality, provenance, safety issues, and usage guidance.
    - Keep the interpretation general—no mandatory dataset-specific statements.

    General instructions:
    - Use bolding for key entities or risks when helpful, but strict Markdown is not required.
    - Keep lists tight while allowing short supporting explanations.
    - Avoid broken formatting or missing links.
    - When multiple IDs are returned from tools, synthesize rather than echo raw outputs.

    Context:
    The agent may incorporate information from the following tools:
    search_huggingface, search_neo4j, extract_training_datasets

    Your goal is to integrate and interpret tool outputs into a clear, actionable, safety-aware report—not to merely restate the raw data.

    Note: Be careful of not hallucinating dataset URLs, since the dataset IDs always contain the author name as prefix, PLEASE ALWAYS use "https://huggingface.co/datasets?search=$dataset" for the dataset link.
    """
    ),
    model="gpt-5.1",
)
