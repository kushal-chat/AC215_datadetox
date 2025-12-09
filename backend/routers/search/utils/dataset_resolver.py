"""Helper utilities for resolving dataset information and HuggingFace links."""

import os
import logging
import re
from typing import Optional, Dict, List
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

logger = logging.getLogger(__name__)

# Initialize HuggingFace API client
hf_token = os.getenv("HF_TOKEN")
hf_api = HfApi(token=hf_token)

# Cache for dataset existence checks to avoid repeated API calls
_dataset_cache: Dict[str, bool] = {}

_VALID_DATASET_ID = re.compile(r"^[\w\-.]+(/[\w\-.]+)?$")


def _looks_like_dataset_id(dataset_id: str) -> bool:
    """Return True if the string looks like a valid HuggingFace dataset identifier."""
    return bool(_VALID_DATASET_ID.match(dataset_id))


def check_dataset_exists(dataset_id: str) -> bool:
    """
    Check if a HuggingFace dataset exists.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "squad", "allenai/c4")

    Returns:
        True if dataset exists, False otherwise
    """
    # Basic sanity check: HuggingFace dataset ids cannot contain whitespace
    if not _looks_like_dataset_id(dataset_id):
        logger.debug(f"Dataset id '{dataset_id}' is not a valid HuggingFace identifier")
        return False

    # Check cache first
    if dataset_id in _dataset_cache:
        return _dataset_cache[dataset_id]

    try:
        hf_api.dataset_info(dataset_id)
        _dataset_cache[dataset_id] = True
        logger.debug(f"Dataset exists: {dataset_id}")
        return True
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            _dataset_cache[dataset_id] = False
            logger.debug(f"Dataset not found: {dataset_id}")
            return False
        logger.warning(f"HTTP error checking dataset {dataset_id}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error checking dataset {dataset_id}: {e}")
        return False


# Well-known dataset mappings (dataset name -> HuggingFace ID)
KNOWN_DATASET_MAPPINGS = {
    # NLP Datasets
    "squad": "rajpurkar/squad",
    "squad_v2": "rajpurkar/squad_v2",
    "glue": "nyu-mll/glue",
    "superglue": "super_glue",
    "wmt": "wmt",
    "bookcorpus": "bookcorpus",
    "wikipedia": "wikipedia",
    "openwebtext": "Skylion007/openwebtext",
    "c4": "allenai/c4",
    "pile": "EleutherAI/pile",
    "redpajama": "togethercomputer/RedPajama-Data-1T",
    # Vision Datasets
    "imagenet": "imagenet-1k",
    "coco": "coco",
    "openimages": "google/open-images",
    "ade20k": "scene_parse_150",
    "cityscapes": "cityscapes",
    "kinetics": "kinetics700",
    "activitynet": "activity_net",
    # Audio Datasets
    "commonvoice": "mozilla-foundation/common_voice_11_0",
    "librispeech": "librispeech_asr",
    "voxceleb": "voxceleb",
    # Multimodal
    "laion": "laion/laion-5b",
    "cc-news": "cc_news",
}


def resolve_dataset_url(
    dataset_name: str, existing_url: Optional[str] = None
) -> Optional[str]:
    """
    Resolve a dataset name to a HuggingFace URL if possible.

    Args:
        dataset_name: Name of the dataset (e.g., "squad", "imagenet")
        existing_url: Existing URL if already known

    Returns:
        HuggingFace dataset URL if found, None otherwise
    """
    # If URL already exists, return it
    if existing_url:
        return existing_url

    dataset_name_lower = dataset_name.lower().strip()

    # Check if it's already a HuggingFace ID format (has a slash)
    if "/" in dataset_name_lower:
        if check_dataset_exists(dataset_name_lower):
            return f"https://huggingface.co/datasets/{dataset_name_lower}"
        return None

    # Check known mappings
    if dataset_name_lower in KNOWN_DATASET_MAPPINGS:
        dataset_id = KNOWN_DATASET_MAPPINGS[dataset_name_lower]
        if check_dataset_exists(dataset_id):
            return f"https://huggingface.co/datasets/{dataset_id}"

    # Try the dataset name directly
    if check_dataset_exists(dataset_name_lower):
        return f"https://huggingface.co/datasets/{dataset_name_lower}"

    # No URL found
    return None


def enrich_dataset_info(datasets: List[Dict]) -> List[Dict]:
    """
    Enrich dataset information with HuggingFace URLs.

    Args:
        datasets: List of dataset dictionaries with 'name', 'url', and 'description'

    Returns:
        Enriched list of datasets with resolved URLs
    """
    enriched = []

    for dataset in datasets:
        name = dataset.get("name")
        url = dataset.get("url")

        enriched_dataset = {
            "name": name,
            "url": url,
            "description": dataset.get("description"),
        }

        # Try to resolve URL if not present
        if name and not url:
            resolved_url = resolve_dataset_url(name, url)
            if resolved_url:
                enriched_dataset["url"] = resolved_url
            else:
                # Add search URL as fallback
                enriched_dataset["hf_search_url"] = (
                    f"https://huggingface.co/datasets?search={name.lower()}"
                )

        enriched.append(enriched_dataset)

    return enriched
