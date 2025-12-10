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


def test_extract_dataset_relationships_from_model(mock_settings, mock_hf_api):
    """Test extracting dataset relationships from model tags."""
    scraper = HuggingFaceScraper()

    mock_model_info = Mock()
    mock_model_info.id = "test/model"

    model_data = {
        "model_id": "test/model",
        "tags": ["nlp", "dataset:squad", "dataset:author/glue"],
    }

    relationships, datasets = scraper._extract_dataset_relationships_from_model(
        mock_model_info, model_data
    )

    assert len(relationships) == 2
    assert len(datasets) == 2
    assert relationships[0]["source"] == "test/model"
    assert relationships[0]["target"] == "squad"
    assert relationships[0]["relationship_type"] == "trained_on"
    assert relationships[0]["target_type"] == "dataset"
    assert datasets[0]["dataset_id"] == "squad"
    assert datasets[1]["dataset_id"] == "author/glue"


def test_extract_dataset_relationships_from_model_no_tags(mock_settings, mock_hf_api):
    """Test extracting dataset relationships when model has no dataset tags."""
    scraper = HuggingFaceScraper()

    mock_model_info = Mock()
    model_data = {"model_id": "test/model", "tags": ["nlp", "bert"]}

    relationships, datasets = scraper._extract_dataset_relationships_from_model(
        mock_model_info, model_data
    )

    assert len(relationships) == 0
    assert len(datasets) == 0


@patch("scrapers.huggingface_scraper.time.sleep")
def test_scrape_datasets(mock_sleep, mock_settings, mock_hf_api):
    """Test scraping datasets."""
    scraper = HuggingFaceScraper()

    # Mock dataset info
    mock_dataset_info = Mock()
    mock_dataset_info.id = "author/dataset1"
    mock_dataset_info.author = "author"
    mock_dataset_info.downloads = 1000
    mock_dataset_info.tags = ["nlp"]
    mock_dataset_info.created_at = None
    mock_dataset_info.updated_at = None

    mock_hf_api.dataset_info.return_value = mock_dataset_info

    with patch.object(
        scraper, "_extract_relationships_from_dataset_card", return_value=[]
    ):
        datasets, relationships = scraper.scrape_datasets(["author/dataset1"])

        assert len(datasets) == 1
        assert datasets[0]["dataset_id"] == "author/dataset1"
        assert datasets[0]["author"] == "author"
        mock_hf_api.dataset_info.assert_called_once_with("author/dataset1")


@patch("scrapers.huggingface_scraper.time.sleep")
def test_scrape_datasets_with_limit(mock_sleep, mock_settings, mock_hf_api):
    """Test scraping datasets with limit."""
    scraper = HuggingFaceScraper()

    mock_dataset_info = Mock()
    mock_dataset_info.id = "author/dataset"
    mock_dataset_info.author = "author"
    mock_dataset_info.downloads = 1000
    mock_dataset_info.tags = []
    mock_dataset_info.created_at = None
    mock_dataset_info.updated_at = None

    mock_hf_api.dataset_info.return_value = mock_dataset_info

    with patch.object(
        scraper, "_extract_relationships_from_dataset_card", return_value=[]
    ):
        datasets, relationships = scraper.scrape_datasets(
            ["author/dataset1", "author/dataset2", "author/dataset3"], limit=2
        )

        # Should only scrape 2 datasets
        assert mock_hf_api.dataset_info.call_count == 2


@patch("scrapers.huggingface_scraper.time.sleep")
def test_scrape_datasets_no_author(mock_sleep, mock_settings, mock_hf_api):
    """Test scraping datasets without author in ID."""
    scraper = HuggingFaceScraper()

    datasets, relationships = scraper.scrape_datasets(["dataset1"])

    # Should skip datasets without author
    assert len(datasets) == 0
    mock_hf_api.dataset_info.assert_not_called()


def test_extract_dataset_info(mock_settings, mock_hf_api):
    """Test extracting dataset information."""
    scraper = HuggingFaceScraper()

    mock_dataset_info = Mock()
    mock_dataset_info.id = "author/dataset"
    mock_dataset_info.author = "author"
    mock_dataset_info.downloads = 1000
    mock_dataset_info.tags = ["nlp", "text"]
    mock_created_at = Mock()
    mock_created_at.isoformat.return_value = "2024-01-01T00:00:00Z"
    mock_updated_at = Mock()
    mock_updated_at.isoformat.return_value = "2024-01-02T00:00:00Z"
    mock_dataset_info.created_at = mock_created_at
    mock_dataset_info.updated_at = mock_updated_at

    dataset_data = scraper._extract_dataset_info(mock_dataset_info)

    assert dataset_data["dataset_id"] == "author/dataset"
    assert dataset_data["author"] == "author"
    assert dataset_data["downloads"] == 1000
    assert dataset_data["tags"] == ["nlp", "text"]
    assert dataset_data["created_at"] == "2024-01-01T00:00:00Z"
    assert dataset_data["updated_at"] == "2024-01-02T00:00:00Z"


def test_extract_dataset_info_none_dates(mock_settings, mock_hf_api):
    """Test extracting dataset info when dates are None."""
    scraper = HuggingFaceScraper()

    mock_dataset_info = Mock()
    mock_dataset_info.id = "author/dataset"
    mock_dataset_info.author = "author"
    mock_dataset_info.downloads = None
    mock_dataset_info.tags = []
    mock_dataset_info.created_at = None
    mock_dataset_info.updated_at = None

    dataset_data = scraper._extract_dataset_info(mock_dataset_info)

    assert dataset_data["created_at"] is None
    assert dataset_data["updated_at"] is None


@patch("scrapers.huggingface_scraper.requests.get")
def test_extract_relationships_from_dataset_card(mock_get, mock_settings, mock_hf_api):
    """Test extracting relationships from dataset card."""
    scraper = HuggingFaceScraper()

    # Mock HTML response with model links
    html_content = """
    <html>
    <body>
        <h2>Models trained or fine-tuned on dataset-name</h2>
        <ul>
            <li><a href="/models/author/model1">model1</a></li>
            <li><a href="/models/author/model2">model2</a></li>
        </ul>
    </body>
    </html>
    """

    mock_response = Mock()
    mock_response.text = html_content
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    relationships = scraper._extract_relationships_from_dataset_card("author/dataset")

    assert len(relationships) == 2
    assert relationships[0]["source"] == "author/model1"
    assert relationships[0]["target"] == "author/dataset"
    assert relationships[0]["relationship_type"] == "trained_on"
    assert relationships[0]["target_type"] == "dataset"
    mock_get.assert_called_once()


@patch("scrapers.huggingface_scraper.requests.get")
def test_extract_relationships_from_dataset_card_no_models(
    mock_get, mock_settings, mock_hf_api
):
    """Test extracting relationships when dataset card has no models section."""
    scraper = HuggingFaceScraper()

    html_content = """
    <html>
    <body>
        <h1>Dataset Card</h1>
        <p>This is a dataset.</p>
    </body>
    </html>
    """

    mock_response = Mock()
    mock_response.text = html_content
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    relationships = scraper._extract_relationships_from_dataset_card("author/dataset")

    assert len(relationships) == 0


@patch("scrapers.huggingface_scraper.requests.get")
def test_get_relationship_type_from_tree(mock_get, mock_settings, mock_hf_api):
    """Test getting relationship type from siblings API."""
    scraper = HuggingFaceScraper()

    # Mock successful API response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "finetuned": [{"id": "child/model"}],
        "adapters": [],
        "merges": [],
        "quantizations": [],
    }
    mock_get.return_value = mock_response

    rel_type = scraper._get_relationship_type_from_tree("child/model", "base/model")

    assert rel_type == "finetuned"
    mock_get.assert_called_once()


@patch("scrapers.huggingface_scraper.requests.get")
def test_get_relationship_type_from_tree_api_fails(
    mock_get, mock_settings, mock_hf_api
):
    """Test getting relationship type when API fails."""
    scraper = HuggingFaceScraper()

    # Mock API failure
    mock_get.side_effect = Exception("API error")

    with patch.object(
        scraper, "_infer_relationship_type_from_name", return_value="finetuned"
    ) as mock_infer:
        rel_type = scraper._get_relationship_type_from_tree("child/model", "base/model")

        assert rel_type == "finetuned"
        mock_infer.assert_called_once()


def test_scrape_model_by_id(mock_settings, mock_hf_api):
    """Test scraping a specific model by ID."""
    scraper = HuggingFaceScraper()

    mock_model_info = Mock()
    mock_model_info.id = "test/model"
    mock_model_info.author = "author"
    mock_model_info.downloads = 1000
    mock_model_info.likes = 50
    mock_model_info.tags = ["nlp"]
    mock_model_info.pipeline_tag = "text-classification"
    mock_model_info.library_name = "transformers"
    mock_model_info.private = False
    mock_model_info.created_at = None
    mock_model_info.last_modified = None
    mock_model_info.sha = "abc123"

    mock_hf_api.model_info.return_value = mock_model_info

    with patch.object(scraper, "_extract_relationships", return_value=[]):
        model_data, relationships = scraper.scrape_model_by_id("test/model")

        assert model_data["model_id"] == "test/model"
        assert len(relationships) == 0
        mock_hf_api.model_info.assert_called_once_with("test/model")


def test_scrape_model_by_id_error(mock_settings, mock_hf_api):
    """Test scraping model by ID when error occurs."""
    scraper = HuggingFaceScraper()

    mock_hf_api.model_info.side_effect = Exception("Model not found")

    with pytest.raises(Exception, match="Model not found"):
        scraper.scrape_model_by_id("nonexistent/model")
