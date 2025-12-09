"""Agent tool for extracting training datasets from arxiv papers."""

import logging
from typing import Dict, Any, List
from agents import function_tool
from .arxiv_extractor import ArxivDatasetExtractor
from .dataset_resolver import enrich_dataset_info
from .tool_state import set_tool_result

logger = logging.getLogger(__name__)


@function_tool
def extract_training_datasets(model_ids: List[str]) -> Dict[str, Any]:
    """
    Extract training dataset information from arxiv papers for given models.

    This tool:
    1. Extracts arxiv paper links from HuggingFace model cards
    2. Fetches and parses the first 8 pages of each arxiv paper (PDF)
    3. Identifies training datasets mentioned in the papers
    4. Returns dataset names and links

    Args:
        model_ids: List of HuggingFace model IDs to process (e.g., ["bert-base-uncased", "gpt2"])

    Returns:
        Dictionary with model_id as key and paper/dataset information as value.
        Each value contains:
        - arxiv_url: Link to the arxiv paper (if found)
        - datasets: List of datasets found, each with:
            - name: Dataset name
            - url: Link to dataset (if available)
            - description: Context around dataset mention (if available)

    Example:
        {
            "bert-base-uncased": {
                "arxiv_url": "https://arxiv.org/abs/1810.04805",
                "datasets": [
                    {
                        "name": "bookcorpus",
                        "url": null,
                        "description": "...trained on BookCorpus and English Wikipedia..."
                    },
                    {
                        "name": "wikipedia",
                        "url": null,
                        "description": "...trained on BookCorpus and English Wikipedia..."
                    }
                ]
            }
        }
    """
    logger.info(f"Extracting training datasets for {len(model_ids)} models")

    try:
        extractor = ArxivDatasetExtractor()

        # Process all models in parallel
        results = extractor.extract_sync(model_ids, max_concurrent=5)

        # Convert to serializable format
        output: Dict[str, Any] = {}
        for model_id, info in results.items():
            datasets = [
                {
                    "name": dataset.name,
                    "url": dataset.url,
                    "description": dataset.description,
                }
                for dataset in info.datasets
            ]

            output[model_id] = {
                "arxiv_url": info.arxiv_url,
                "datasets": enrich_dataset_info(datasets),
            }

        logger.info(f"Successfully extracted datasets for {len(output)} models")

        # Store the result in request-scoped state for later retrieval
        set_tool_result("extract_training_datasets", output)

        return output

    except Exception as e:
        logger.error(f"Error extracting training datasets: {e}")
        error_result = {
            "error": str(e),
            "message": "Failed to extract training datasets",
        }
        set_tool_result("extract_training_datasets", error_result)
        return error_result
