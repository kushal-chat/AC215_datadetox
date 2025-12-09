"""Utilities for deriving dataset risk context before LLM analysis."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List

SYNTHETIC_KEYWORDS = {
    "synthetic",
    "generated",
    "model-generated",
}

ENGLISH_KEYWORDS = {
    "english",
    "en",
    "uk",
    "us",
    "american",
}

HIGH_RISK_DATASETS = {
    "pile",
    "redpajama",
}


def _normalize(text: str | None) -> str:
    return text.lower() if text else ""


def _flag_synthetic(text: str) -> bool:
    return any(keyword in text for keyword in SYNTHETIC_KEYWORDS)


def _flag_english_bias(text: str) -> bool:
    if not text:
        return False
    # ensure keywords surrounded by word boundaries to reduce false positives
    return any(
        re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in ENGLISH_KEYWORDS
    )


def _dataset_risk(dataset: Dict[str, Any]) -> Dict[str, Any]:
    name = dataset.get("name", "unknown")
    desc = _normalize(dataset.get("description"))
    url = dataset.get("url")

    indicators: List[str] = []
    score = 0

    if _flag_synthetic(_normalize(name) + " " + desc):
        indicators.append("synthetic_source")
        score += 2

    if _flag_english_bias(_normalize(name)) or _flag_english_bias(desc):
        indicators.append("english_centric")
        score += 1

    if not url:
        indicators.append("no_verified_source")
        score += 1

    if _normalize(name) in HIGH_RISK_DATASETS:
        indicators.append("known_large_crawl")
        score += 1

    if not indicators:
        indicators.append("no_specific_flags")

    risk_level = "low"
    if score >= 3:
        risk_level = "high"
    elif score >= 1:
        risk_level = "medium"

    return {
        "name": name,
        "risk_level": risk_level,
        "indicators": indicators,
        "url_present": bool(url),
    }


def build_dataset_risk_context(
    training_dataset_map: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Summarize dataset risks per model for downstream LLM analysis."""

    context: Dict[str, Any] = {
        "models": [],
        "global_counts": {
            "high": 0,
            "medium": 0,
            "low": 0,
            "unknown_models": 0,
        },
    }

    if not isinstance(training_dataset_map, dict):
        return context

    risk_counter = Counter()

    for model_id, info in training_dataset_map.items():
        datasets = info.get("datasets") or []
        if not datasets:
            context["global_counts"]["unknown_models"] += 1
            continue

        model_entry = {
            "model_id": model_id,
            "arxiv_url": info.get("arxiv_url"),
            "datasets": [],
        }

        for dataset in datasets:
            assessment = _dataset_risk(dataset)
            model_entry["datasets"].append(assessment)
            risk_counter[assessment["risk_level"]] += 1

        context["models"].append(model_entry)

    context["global_counts"].update(risk_counter)
    return context
