import os
import logging
from typing import Optional, Dict, List
from agents import function_tool
from huggingface_hub import HfApi, ModelCard, DatasetCard
from huggingface_hub.utils import HfHubHTTPError

logger = logging.getLogger(__name__)

# Initialize HuggingFace API client
hf_token = os.getenv("HF_TOKEN")
hf_api = HfApi(token=hf_token)

def search_models(
    query: str, limit: int = 5, sort: str = "downloads"
) -> List[Dict[str, str]]:
    """
    Search for models on HuggingFace Hub.

    Args:
        query: Search query string
        limit: Maximum number of results
        sort: Sort by 'downloads', 'likes', 'created', or 'modified'

    Returns:
        List of model info dictionaries
    """
    try:
        logger.info(f"Searching HuggingFace models for: {query}")

        # Search models
        models = hf_api.list_models(
            search=query,
            limit=limit,
            sort=sort,
            direction=-1,  # Descending order
        )

        results = []
        for model in models:
            model_info = {
                "id": model.id,
                "author": model.author or "Unknown",
                "downloads": model.downloads or 0,
                "likes": model.likes or 0,
                "tags": model.tags or [],
                "pipeline_tag": model.pipeline_tag or "Unknown",
                "url": f"https://huggingface.co/{model.id}",
                "created_at": str(model.created_at) if model.created_at else "Unknown",
                "last_modified": (
                    str(model.last_modified) if model.last_modified else "Unknown"
                ),
            }
            results.append(model_info)

        logger.info(f"Found {len(results)} models")
        return results

    except Exception as e:
        logger.error(f"Error searching models: {str(e)}")
        return []


def search_datasets(
    query: str, limit: int = 5, sort: str = "downloads"
) -> List[Dict[str, str]]:
    """
    Search for datasets on HuggingFace Hub.

    Args:
        query: Search query string
        limit: Maximum number of results
        sort: Sort by 'downloads', 'likes', 'created', or 'modified'

    Returns:
        List of dataset info dictionaries
    """
    try:
        logger.info(f"Searching HuggingFace datasets for: {query}")

        # Search datasets
        datasets = hf_api.list_datasets(
            search=query,
            limit=limit,
            sort=sort,
            direction=-1,  # Descending order
        )

        results = []
        for dataset in datasets:
            dataset_info = {
                "id": dataset.id,
                "author": dataset.author or "Unknown",
                "downloads": dataset.downloads or 0,
                "likes": dataset.likes or 0,
                "tags": dataset.tags or [],
                "url": f"https://huggingface.co/datasets/{dataset.id}",
                "created_at": (
                    str(dataset.created_at) if dataset.created_at else "Unknown"
                ),
                "last_modified": (
                    str(dataset.last_modified) if dataset.last_modified else "Unknown"
                ),
            }
            results.append(dataset_info)

        logger.info(f"Found {len(results)} datasets")
        return results

    except Exception as e:
        logger.error(f"Error searching datasets: {str(e)}")
        return []


def get_model_card(model_id: str) -> Optional[Dict[str, str]]:
    """
    Fetch model card information from HuggingFace.

    Args:
        model_id: HuggingFace model ID (e.g., "bert-base-uncased")

    Returns:
        Dict with model card info or None if not found
    """
    try:
        logger.info(f"Fetching model card for: {model_id}")

        # Get model info
        model_info = hf_api.model_info(model_id)

        # Try to get model card text
        try:
            card = ModelCard.load(model_id)
            card_text = card.text
        except Exception as e:
            logger.warning(f"Could not load model card text: {e}")
            card_text = None

        result = {
            "id": model_info.id,
            "author": model_info.author or "Unknown",
            "downloads": model_info.downloads or 0,
            "likes": model_info.likes or 0,
            "tags": model_info.tags or [],
            "pipeline_tag": model_info.pipeline_tag or "Unknown",
            "library_name": model_info.library_name or "Unknown",
            "url": f"https://huggingface.co/{model_info.id}",
            "card_text": card_text or "Model card not available",
            "created_at": (
                str(model_info.created_at) if model_info.created_at else "Unknown"
            ),
            "last_modified": (
                str(model_info.last_modified) if model_info.last_modified else "Unknown"
            ),
        }

        logger.info(f"Successfully fetched model card for {model_id}")
        return result

    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            logger.error(f"Model not found: {model_id}")
        else:
            logger.error(f"HTTP error fetching model {model_id}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error fetching model card: {str(e)}")
        return None


def get_dataset_card(dataset_id: str) -> Optional[Dict[str, str]]:
    """
    Fetch dataset card information from HuggingFace.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "squad")

    Returns:
        Dict with dataset card info or None if not found
    """
    try:
        logger.info(f"Fetching dataset card for: {dataset_id}")

        # Get dataset info
        dataset_info = hf_api.dataset_info(dataset_id)

        # Try to get dataset card text
        try:
            card = DatasetCard.load(dataset_id)
            card_text = card.text
        except Exception as e:
            logger.warning(f"Could not load dataset card text: {e}")
            card_text = None

        result = {
            "id": dataset_info.id,
            "author": dataset_info.author or "Unknown",
            "downloads": dataset_info.downloads or 0,
            "likes": dataset_info.likes or 0,
            "tags": dataset_info.tags or [],
            "url": f"https://huggingface.co/datasets/{dataset_info.id}",
            "card_text": card_text or "Dataset card not available",
            "created_at": (
                str(dataset_info.created_at) if dataset_info.created_at else "Unknown"
            ),
            "last_modified": (
                str(dataset_info.last_modified)
                if dataset_info.last_modified
                else "Unknown"
            ),
        }

        logger.info(f"Successfully fetched dataset card for {dataset_id}")
        return result

    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            logger.error(f"Dataset not found: {dataset_id}")
        else:
            logger.error(f"HTTP error fetching dataset {dataset_id}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error fetching dataset card: {str(e)}")
        return None


def format_search_results(models: List[Dict], datasets: List[Dict]) -> str:
    """
    Format search results into a readable string for LLM consumption.

    Args:
        models: List of model info dicts
        datasets: List of dataset info dicts

    Returns:
        Formatted string with search results
    """
    output = []

    if models:
        output.append("### 🤖 Models Found:\n")
        for i, model in enumerate(models, 1):
            output.append(f"**{i}. [{model['id']}]({model['url']})**")
            output.append(f"   - Author: {model['author']}")
            output.append(f"   - Downloads: {model['downloads']:,}")
            output.append(f"   - Likes: {model['likes']}")
            output.append(f"   - Task: {model['pipeline_tag']}")
            output.append(f"   - Tags: {', '.join(model['tags'][:5])}")
            output.append("")

    if datasets:
        output.append("\n### 📊 Datasets Found:\n")
        for i, dataset in enumerate(datasets, 1):
            output.append(f"**{i}. [{dataset['id']}]({dataset['url']})**")
            output.append(f"   - Author: {dataset['author']}")
            output.append(f"   - Downloads: {dataset['downloads']:,}")
            output.append(f"   - Likes: {dataset['likes']}")
            output.append(f"   - Tags: {', '.join(dataset['tags'][:5])}")
            output.append("")

    if not models and not datasets:
        output.append("No results found on HuggingFace Hub.")

    return "\n".join(output)


def search_huggingface_function(
    query: str, include_models: bool = True, include_datasets: bool = True
) -> str:
    """
    Search HuggingFace Hub for models and/or datasets.

    Args:
        query: Search query
        include_models: Whether to search models
        include_datasets: Whether to search datasets

    Returns:
        Formatted search results
    """
    models = []
    datasets = []

    logger.info("HuggingFace tool has been called.")

    if include_models:
        models = search_models(query, limit=3)

    if include_datasets:
        datasets = search_datasets(query, limit=3)

    return format_search_results(models, datasets)


@function_tool
def search_huggingface(
    query: str, include_models: bool = True, include_datasets: bool = True
) -> str:
    """
    Function tool version of search_huggingface.
    """
    return search_huggingface_function(
        query, include_models=include_models, include_datasets=include_datasets
    )
