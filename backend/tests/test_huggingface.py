import types
from typing import List, Dict

import pytest

from routers.search.utils import huggingface


# ---------- Helpers for dummy objects ----------
# Many of these tests are LLM generated to have random
# information feasible for large-scale testing


class DummyModel:
    def __init__(
        self,
        _id: str = "user/model",
        author: str = "author",
        downloads: int = 1234,
        likes: int = 10,
        tags: List[str] | None = None,
        pipeline_tag: str | None = "text-classification",
        created_at: str | None = "2024-01-01T00:00:00",
        last_modified: str | None = "2024-02-01T00:00:00",
    ) -> None:
        self.id = _id
        self.author = author
        self.downloads = downloads
        self.likes = likes
        self.tags = tags or ["tag1", "tag2"]
        self.pipeline_tag = pipeline_tag
        self.created_at = created_at
        self.last_modified = last_modified


class DummyDataset:
    def __init__(
        self,
        _id: str = "user/dataset",
        author: str = "author",
        downloads: int = 5678,
        likes: int = 5,
        tags: List[str] | None = None,
        created_at: str | None = "2024-01-01T00:00:00",
        last_modified: str | None = "2024-02-01T00:00:00",
    ) -> None:
        self.id = _id
        self.author = author
        self.downloads = downloads
        self.likes = likes
        self.tags = tags or ["tagA", "tagB"]
        self.created_at = created_at
        self.last_modified = last_modified


# ---------- search_models ----------


def test_search_models_success(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_models = [
        DummyModel(_id="a/model-1", downloads=1000),
        DummyModel(_id="b/model-2", downloads=2000),
    ]

    def fake_list_models(*, search: str, limit: int, sort: str, direction: int):
        assert search == "bert"
        assert limit == 3
        assert sort == "downloads"
        assert direction == -1
        return dummy_models

    monkeypatch.setattr(
        huggingface.hf_api, "list_models", fake_list_models, raising=False
    )

    results = huggingface.search_models("bert", limit=3, sort="downloads")

    assert len(results) == 2
    assert results[0]["id"] == "a/model-1"
    assert results[0]["downloads"] == 1000
    assert results[0]["url"] == "https://huggingface.co/a/model-1"
    assert results[1]["id"] == "b/model-2"


def test_search_models_error_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_list_models(*, search: str, limit: int, sort: str, direction: int):
        raise RuntimeError("Boom")

    monkeypatch.setattr(
        huggingface.hf_api, "list_models", fake_list_models, raising=False
    )

    results = huggingface.search_models("anything")
    assert results == []


# ---------- search_datasets ----------


def test_search_datasets_success(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_datasets = [
        DummyDataset(_id="user/dataset-1", downloads=111),
        DummyDataset(_id="user/dataset-2", downloads=222),
    ]

    def fake_list_datasets(*, search: str, limit: int, sort: str, direction: int):
        assert search == "wmt"
        assert limit == 2
        assert sort == "downloads"
        assert direction == -1
        return dummy_datasets

    monkeypatch.setattr(
        huggingface.hf_api, "list_datasets", fake_list_datasets, raising=False
    )

    results = huggingface.search_datasets("wmt", limit=2, sort="downloads")

    assert len(results) == 2
    assert results[0]["id"] == "user/dataset-1"
    assert results[0]["url"] == "https://huggingface.co/datasets/user/dataset-1"
    assert results[1]["downloads"] == 222


def test_search_datasets_error_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_list_datasets(*, search: str, limit: int, sort: str, direction: int):
        raise RuntimeError("Boom")

    monkeypatch.setattr(
        huggingface.hf_api, "list_datasets", fake_list_datasets, raising=False
    )

    results = huggingface.search_datasets("anything")
    assert results == []


def test_get_model_card_error_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_model_info(model_id: str):
        raise RuntimeError("HTTP 500")

    monkeypatch.setattr(
        huggingface.hf_api, "model_info", fake_model_info, raising=False
    )

    result = huggingface.get_model_card("does-not-exist")
    assert result is None


# ---------- get_dataset_card ----------


def test_get_dataset_card_success_with_card_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = DummyDataset(_id="user/dataset-card", downloads=999)

    def fake_dataset_info(dataset_id: str):
        assert dataset_id == "user/dataset-card"
        return info

    def fake_load(repo_id: str):
        assert repo_id == "user/dataset-card"
        return types.SimpleNamespace(text="This is a dataset card")

    monkeypatch.setattr(
        huggingface.hf_api, "dataset_info", fake_dataset_info, raising=False
    )
    dummy_dataset_card = types.SimpleNamespace(load=fake_load)
    monkeypatch.setattr(huggingface, "DatasetCard", dummy_dataset_card, raising=False)

    result = huggingface.get_dataset_card("user/dataset-card")
    assert result is not None
    assert result["id"] == "user/dataset-card"
    assert result["card_text"] == "This is a dataset card"
    assert result["url"] == "https://huggingface.co/datasets/user/dataset-card"
    assert result["downloads"] == 999


def test_get_dataset_card_card_text_failure_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = DummyDataset(_id="user/dataset-card-no-text")

    def fake_dataset_info(dataset_id: str):
        return info

    def fake_load(_repo_id: str):
        raise RuntimeError("cannot load card")

    monkeypatch.setattr(
        huggingface.hf_api, "dataset_info", fake_dataset_info, raising=False
    )
    dummy_dataset_card = types.SimpleNamespace(load=fake_load)
    monkeypatch.setattr(huggingface, "DatasetCard", dummy_dataset_card, raising=False)

    result = huggingface.get_dataset_card("user/dataset-card-no-text")
    assert result is not None
    assert result["card_text"] == "Dataset card not available"


def test_get_dataset_card_error_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_dataset_info(dataset_id: str):
        raise RuntimeError("HTTP 500")

    monkeypatch.setattr(
        huggingface.hf_api, "dataset_info", fake_dataset_info, raising=False
    )

    result = huggingface.get_dataset_card("does-not-exist")
    assert result is None


# ---------- search_huggingface (function_tool wrapper) ----------


def test_search_huggingface_calls_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_search_models(query: str, limit: int = 5, sort: str = "downloads"):
        assert query == "llama"
        assert limit == 3  # wrapper passes 3
        assert sort == "downloads"
        return [
            {
                "id": "user/llama",
                "author": "meta",
                "downloads": 123,
                "likes": 4,
                "tags": [],
                "pipeline_tag": "text-generation",
                "url": "https://huggingface.co/user/llama",
                "created_at": "2024-01-01",
                "last_modified": "2024-01-02",
            }
        ]

    def fake_search_datasets(query: str, limit: int = 5, sort: str = "downloads"):
        assert query == "llama"
        assert limit == 3
        assert sort == "downloads"
        return []

    monkeypatch.setattr(huggingface, "search_models", fake_search_models, raising=False)
    monkeypatch.setattr(
        huggingface, "search_datasets", fake_search_datasets, raising=False
    )

    result_str = huggingface.search_huggingface_function(
        "llama", include_models=True, include_datasets=True
    )

    assert "user/llama" in result_str
    assert "Models Found" in result_str
