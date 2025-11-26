"""Tests for HuggingFace scraper."""

from __future__ import annotations

from unittest.mock import Mock, MagicMock, patch
import pytest

from scrapers.huggingface_scraper import HuggingFaceScraper


@pytest.fixture
def mock_settings():
    """Mock settings."""
    with patch("scrapers.huggingface_scraper.settings") as mock_settings:
        mock_settings.HF_TOKEN = "test_token"
        mock_settings.RATE_LIMIT_DELAY = 0.1
        mock_settings.validate.return_value = None
        yield mock_settings


@pytest.fixture
def mock_hf_api():
    """Mock HuggingFace API."""
    api = MagicMock()
    with patch("scrapers.huggingface_scraper.HfApi", return_value=api):
        yield api


def test_scraper_init(mock_settings, mock_hf_api):
    """Test scraper initialization."""
    scraper = HuggingFaceScraper()
    assert scraper.api == mock_hf_api
    assert scraper.rate_limit_delay == 0.1


def test_extract_model_info(mock_settings, mock_hf_api):
    """Test extracting model information."""
    scraper = HuggingFaceScraper()

    mock_model_info = Mock()
    mock_model_info.id = "test/model"
    mock_model_info.author = "test_author"
    mock_model_info.downloads = 1000
    mock_model_info.likes = 50
    mock_model_info.tags = ["nlp", "bert"]
    mock_model_info.pipeline_tag = "text-classification"
    mock_model_info.library_name = "transformers"
    mock_model_info.private = False
    mock_created_at = Mock()
    mock_created_at.isoformat.return_value = "2024-01-01T00:00:00Z"
    mock_updated_at = Mock()
    mock_updated_at.isoformat.return_value = "2024-01-02T00:00:00Z"
    mock_model_info.created_at = mock_created_at
    mock_model_info.updated_at = mock_updated_at
    mock_model_info.last_modified = mock_updated_at
    mock_model_info.sha = "abc123"

    model_data = scraper._extract_model_info(mock_model_info)

    assert model_data["model_id"] == "test/model"
    assert model_data["author"] == "test_author"
    assert model_data["downloads"] == 1000
    assert model_data["likes"] == 50
    assert model_data["tags"] == ["nlp", "bert"]
    assert model_data["pipeline_tag"] == "text-classification"
    assert model_data["library_name"] == "transformers"
    assert model_data["private"] is False
    assert "url" in model_data


def test_extract_model_info_with_none_values(mock_settings, mock_hf_api):
    """Test extracting model info when some fields are None."""
    scraper = HuggingFaceScraper()

    mock_model_info = Mock()
    mock_model_info.id = "test/model"
    mock_model_info.author = None
    mock_model_info.downloads = None
    mock_model_info.likes = None
    mock_model_info.tags = []
    mock_model_info.pipeline_tag = None
    mock_model_info.library_name = None
    mock_model_info.private = False
    mock_model_info.created_at = None
    mock_model_info.last_modified = None
    mock_model_info.sha = "abc123"

    model_data = scraper._extract_model_info(mock_model_info)

    assert model_data["model_id"] == "test/model"
    assert model_data["author"] is None
    assert model_data["downloads"] is None
    assert "url" in model_data


@patch("huggingface_hub.ModelCard")
def test_get_base_model_from_card(mock_model_card, mock_settings, mock_hf_api):
    """Test getting base model from model card."""
    scraper = HuggingFaceScraper()

    # Mock ModelCard with base_model in data
    mock_card = Mock()
    mock_card.data = {"base_model": "base/model"}
    mock_model_card.load.return_value = mock_card

    mock_model_info = Mock()
    mock_model_info.id = "test/model"
    mock_model_info.author = "test_author"

    base_model = scraper._get_base_model_from_card(mock_model_info)

    assert base_model == "base/model"
    mock_model_card.load.assert_called_once_with("test/model")


@patch("huggingface_hub.ModelCard")
def test_get_base_model_from_card_no_base_model(
    mock_model_card, mock_settings, mock_hf_api
):
    """Test getting base model when card has no base_model field."""
    scraper = HuggingFaceScraper()

    # Mock ModelCard without base_model
    mock_card = Mock()
    mock_card.data = {}
    mock_model_card.load.return_value = mock_card

    mock_model_info = Mock()
    mock_model_info.id = "test/model"
    mock_model_info.author = "test_author"

    base_model = scraper._get_base_model_from_card(mock_model_info)

    assert base_model is None


def test_extract_relationships_based_on(mock_settings, mock_hf_api):
    """Test extracting relationship."""
    scraper = HuggingFaceScraper()

    mock_model_info = Mock()
    mock_model_info.id = "child/model"

    model_data = {"model_id": "child/model"}

    with (
        patch.object(scraper, "_get_base_model_from_card", return_value="parent/model"),
        patch.object(
            scraper, "_get_relationship_type_from_tree", return_value="finetuned"
        ),
    ):
        relationships = scraper._extract_relationships(mock_model_info, model_data)

        assert len(relationships) == 1
        assert relationships[0]["source"] == "child/model"
        assert relationships[0]["target"] == "parent/model"
        assert relationships[0]["relationship_type"] == "finetuned"
        assert relationships[0]["source_type"] == "model"
        assert relationships[0]["target_type"] == "model"


def test_extract_relationships_no_base_model(mock_settings, mock_hf_api):
    """Test extracting relationships when no base model exists."""
    scraper = HuggingFaceScraper()

    mock_model_info = Mock()
    mock_model_info.id = "standalone/model"

    model_data = {"model_id": "standalone/model"}

    with patch.object(scraper, "_get_base_model_from_card", return_value=None):
        relationships = scraper._extract_relationships(mock_model_info, model_data)

        assert len(relationships) == 0


def test_infer_relationship_type_from_name(mock_settings, mock_hf_api):
    """Test inferring relationship type from model names."""
    scraper = HuggingFaceScraper()

    # Test quantization pattern - pattern is in model_id (first param)
    rel_type = scraper._infer_relationship_type_from_name(
        "base-model-4bit", "base-model"
    )
    assert rel_type == "quantizations"

    # Test adapter pattern
    rel_type = scraper._infer_relationship_type_from_name(
        "base-model-lora", "base-model"
    )
    assert rel_type == "adapters"

    # Test merge pattern
    rel_type = scraper._infer_relationship_type_from_name(
        "base-model-merge", "base-model"
    )
    assert rel_type == "merges"

    # Test default (finetuned when base_model exists and no pattern matches)
    rel_type = scraper._infer_relationship_type_from_name(
        "base-model-variant", "base-model"
    )
    assert rel_type == "finetuned"

    # Test None when model_id equals base_model (no relationship)
    rel_type = scraper._infer_relationship_type_from_name("base-model", "base-model")
    assert rel_type is None

    # Test None when base_model is empty (no base model)
    rel_type = scraper._infer_relationship_type_from_name("standalone-model", "")
    assert rel_type is None
