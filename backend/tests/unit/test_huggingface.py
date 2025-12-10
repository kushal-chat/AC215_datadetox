"""Unit tests for huggingface.py functions."""

from unittest.mock import Mock, patch
from huggingface_hub.utils import HfHubHTTPError
from routers.search.utils.huggingface import (
    search_models,
    search_datasets,
    get_model_card,
    get_dataset_card,
    format_search_results,
    search_huggingface_function,
)


class TestSearchModels:
    """Tests for search_models function."""

    @patch("routers.search.utils.huggingface.hf_api")
    def test_search_models_success(self, mock_api):
        """Test successful model search."""
        # Mock model objects
        mock_model1 = Mock()
        mock_model1.id = "model1/test"
        mock_model1.author = "author1"
        mock_model1.downloads = 1000
        mock_model1.likes = 50
        mock_model1.tags = ["nlp", "transformer"]
        mock_model1.pipeline_tag = "text-generation"
        mock_model1.created_at = "2024-01-01"
        mock_model1.last_modified = "2024-01-02"

        mock_model2 = Mock()
        mock_model2.id = "model2/test"
        mock_model2.author = "author2"
        mock_model2.downloads = 2000
        mock_model2.likes = 100
        mock_model2.tags = ["vision"]
        mock_model2.pipeline_tag = "image-classification"
        mock_model2.created_at = "2024-01-03"
        mock_model2.last_modified = "2024-01-04"

        mock_api.list_models.return_value = [mock_model1, mock_model2]

        result = search_models("test query", limit=5)
        assert len(result) == 2
        assert result[0]["id"] == "model1/test"
        assert result[0]["downloads"] == 1000
        assert result[1]["id"] == "model2/test"
        mock_api.list_models.assert_called_once()

    @patch("routers.search.utils.huggingface.hf_api")
    def test_search_models_empty(self, mock_api):
        """Test model search with no results."""
        mock_api.list_models.return_value = []
        result = search_models("test query")
        assert result == []

    @patch("routers.search.utils.huggingface.hf_api")
    def test_search_models_handles_none_values(self, mock_api):
        """Test that None values are handled gracefully."""
        mock_model = Mock()
        mock_model.id = "model/test"
        mock_model.author = None
        mock_model.downloads = None
        mock_model.likes = None
        mock_model.tags = None
        mock_model.pipeline_tag = None
        mock_model.created_at = None
        mock_model.last_modified = None

        mock_api.list_models.return_value = [mock_model]

        result = search_models("test")
        assert len(result) == 1
        assert result[0]["author"] == "Unknown"
        assert result[0]["downloads"] == 0
        assert result[0]["likes"] == 0

    @patch("routers.search.utils.huggingface.hf_api")
    def test_search_models_exception(self, mock_api):
        """Test that exceptions are caught and empty list returned."""
        mock_api.list_models.side_effect = Exception("API error")
        result = search_models("test")
        assert result == []


class TestSearchDatasets:
    """Tests for search_datasets function."""

    @patch("routers.search.utils.huggingface.hf_api")
    def test_search_datasets_success(self, mock_api):
        """Test successful dataset search."""
        mock_dataset1 = Mock()
        mock_dataset1.id = "dataset1/test"
        mock_dataset1.author = "author1"
        mock_dataset1.downloads = 500
        mock_dataset1.likes = 25
        mock_dataset1.tags = ["nlp"]
        mock_dataset1.created_at = "2024-01-01"
        mock_dataset1.last_modified = "2024-01-02"

        mock_api.list_datasets.return_value = [mock_dataset1]

        result = search_datasets("test query", limit=5)
        assert len(result) == 1
        assert result[0]["id"] == "dataset1/test"
        assert result[0]["downloads"] == 500

    @patch("routers.search.utils.huggingface.hf_api")
    def test_search_datasets_empty(self, mock_api):
        """Test dataset search with no results."""
        mock_api.list_datasets.return_value = []
        result = search_datasets("test query")
        assert result == []

    @patch("routers.search.utils.huggingface.hf_api")
    def test_search_datasets_exception(self, mock_api):
        """Test that exceptions are caught and empty list returned."""
        mock_api.list_datasets.side_effect = Exception("API error")
        result = search_datasets("test")
        assert result == []


class TestGetModelCard:
    """Tests for get_model_card function."""

    @patch("routers.search.utils.huggingface.ModelCard")
    @patch("routers.search.utils.huggingface.hf_api")
    def test_get_model_card_success(self, mock_api, mock_model_card_class):
        """Test successful model card retrieval."""
        mock_model_info = Mock()
        mock_model_info.id = "model/test"
        mock_model_info.author = "author"
        mock_model_info.downloads = 1000
        mock_model_info.likes = 50
        mock_model_info.tags = ["nlp"]
        mock_model_info.pipeline_tag = "text-generation"
        mock_model_info.library_name = "transformers"
        mock_model_info.created_at = "2024-01-01"
        mock_model_info.last_modified = "2024-01-02"

        mock_api.model_info.return_value = mock_model_info

        mock_card = Mock()
        mock_card.text = "Model card text"
        mock_model_card_class.load.return_value = mock_card

        result = get_model_card("model/test")
        assert result is not None
        assert result["id"] == "model/test"
        assert result["card_text"] == "Model card text"

    @patch("routers.search.utils.huggingface.ModelCard")
    @patch("routers.search.utils.huggingface.hf_api")
    def test_get_model_card_no_card_text(self, mock_api, mock_model_card_class):
        """Test model card retrieval when card text is unavailable."""
        mock_model_info = Mock()
        mock_model_info.id = "model/test"
        mock_model_info.author = "author"
        mock_model_info.downloads = 1000
        mock_model_info.likes = 50
        mock_model_info.tags = []
        mock_model_info.pipeline_tag = "text-generation"
        mock_model_info.library_name = "transformers"
        mock_model_info.created_at = "2024-01-01"
        mock_model_info.last_modified = "2024-01-02"

        mock_api.model_info.return_value = mock_model_info
        mock_model_card_class.load.side_effect = Exception("Card not found")

        result = get_model_card("model/test")
        assert result is not None
        assert result["card_text"] == "Model card not available"

    @patch("routers.search.utils.huggingface.hf_api")
    def test_get_model_card_404(self, mock_api):
        """Test model card retrieval with 404 error."""
        mock_error = HfHubHTTPError("Not found")
        mock_error.response = Mock()
        mock_error.response.status_code = 404
        mock_api.model_info.side_effect = mock_error

        result = get_model_card("nonexistent/model")
        assert result is None

    @patch("routers.search.utils.huggingface.hf_api")
    def test_get_model_card_other_http_error(self, mock_api):
        """Test model card retrieval with other HTTP error."""
        mock_error = HfHubHTTPError("Server error")
        mock_error.response = Mock()
        mock_error.response.status_code = 500
        mock_api.model_info.side_effect = mock_error

        result = get_model_card("model/test")
        assert result is None

    @patch("routers.search.utils.huggingface.hf_api")
    def test_get_model_card_general_exception(self, mock_api):
        """Test model card retrieval with general exception."""
        mock_api.model_info.side_effect = Exception("Unexpected error")
        result = get_model_card("model/test")
        assert result is None


class TestGetDatasetCard:
    """Tests for get_dataset_card function."""

    @patch("routers.search.utils.huggingface.DatasetCard")
    @patch("routers.search.utils.huggingface.hf_api")
    def test_get_dataset_card_success(self, mock_api, mock_dataset_card_class):
        """Test successful dataset card retrieval."""
        mock_dataset_info = Mock()
        mock_dataset_info.id = "dataset/test"
        mock_dataset_info.author = "author"
        mock_dataset_info.downloads = 500
        mock_dataset_info.likes = 25
        mock_dataset_info.tags = ["nlp"]
        mock_dataset_info.created_at = "2024-01-01"
        mock_dataset_info.last_modified = "2024-01-02"

        mock_api.dataset_info.return_value = mock_dataset_info

        mock_card = Mock()
        mock_card.text = "Dataset card text"
        mock_dataset_card_class.load.return_value = mock_card

        result = get_dataset_card("dataset/test")
        assert result is not None
        assert result["id"] == "dataset/test"
        assert result["card_text"] == "Dataset card text"

    @patch("routers.search.utils.huggingface.DatasetCard")
    @patch("routers.search.utils.huggingface.hf_api")
    def test_get_dataset_card_no_card_text(self, mock_api, mock_dataset_card_class):
        """Test dataset card retrieval when card text is unavailable."""
        mock_dataset_info = Mock()
        mock_dataset_info.id = "dataset/test"
        mock_dataset_info.author = "author"
        mock_dataset_info.downloads = 500
        mock_dataset_info.likes = 25
        mock_dataset_info.tags = []
        mock_dataset_info.created_at = "2024-01-01"
        mock_dataset_info.last_modified = "2024-01-02"

        mock_api.dataset_info.return_value = mock_dataset_info
        mock_dataset_card_class.load.side_effect = Exception("Card not found")

        result = get_dataset_card("dataset/test")
        assert result is not None
        assert result["card_text"] == "Dataset card not available"

    @patch("routers.search.utils.huggingface.hf_api")
    def test_get_dataset_card_404(self, mock_api):
        """Test dataset card retrieval with 404 error."""
        mock_error = HfHubHTTPError("Not found")
        mock_error.response = Mock()
        mock_error.response.status_code = 404
        mock_api.dataset_info.side_effect = mock_error

        result = get_dataset_card("nonexistent/dataset")
        assert result is None


class TestFormatSearchResults:
    """Tests for format_search_results function."""

    def test_format_with_models_and_datasets(self):
        """Test formatting with both models and datasets."""
        models = [
            {
                "id": "model1/test",
                "author": "author1",
                "downloads": 1000,
                "likes": 50,
                "pipeline_tag": "text-generation",
                "tags": ["nlp", "transformer"],
                "url": "https://huggingface.co/model1/test",
            }
        ]
        datasets = [
            {
                "id": "dataset1/test",
                "author": "author2",
                "downloads": 500,
                "likes": 25,
                "tags": ["nlp"],
                "url": "https://huggingface.co/datasets/dataset1/test",
            }
        ]

        result = format_search_results(models, datasets)
        assert "Models Found" in result
        assert "Datasets Found" in result
        assert "model1/test" in result
        assert "dataset1/test" in result

    def test_format_with_models_only(self):
        """Test formatting with only models."""
        models = [
            {
                "id": "model1/test",
                "author": "author1",
                "downloads": 1000,
                "likes": 50,
                "pipeline_tag": "text-generation",
                "tags": ["nlp"],
                "url": "https://huggingface.co/model1/test",
            }
        ]
        result = format_search_results(models, [])
        assert "Models Found" in result
        assert "Datasets Found" not in result

    def test_format_with_datasets_only(self):
        """Test formatting with only datasets."""
        datasets = [
            {
                "id": "dataset1/test",
                "author": "author2",
                "downloads": 500,
                "likes": 25,
                "tags": ["nlp"],
                "url": "https://huggingface.co/datasets/dataset1/test",
            }
        ]
        result = format_search_results([], datasets)
        assert "Models Found" not in result
        assert "Datasets Found" in result

    def test_format_empty(self):
        """Test formatting with no results."""
        result = format_search_results([], [])
        assert "No results found" in result

    def test_format_limits_tags(self):
        """Test that tags are limited to 5."""
        models = [
            {
                "id": "model1/test",
                "author": "author1",
                "downloads": 1000,
                "likes": 50,
                "pipeline_tag": "text-generation",
                "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"],
                "url": "https://huggingface.co/model1/test",
            }
        ]
        result = format_search_results(models, [])
        # Should only show first 5 tags (code uses [:5])
        # Count tags by splitting on comma
        tag_line = [line for line in result.split("\n") if "Tags:" in line][0]
        tags_shown = tag_line.split("Tags:")[1].strip().split(", ")
        assert len(tags_shown) <= 5


class TestSearchHuggingfaceFunction:
    """Tests for search_huggingface_function."""

    @patch("routers.search.utils.huggingface.search_models")
    @patch("routers.search.utils.huggingface.search_datasets")
    @patch("routers.search.utils.huggingface.format_search_results")
    def test_search_both_models_and_datasets(
        self, mock_format, mock_search_datasets, mock_search_models
    ):
        """Test searching for both models and datasets."""
        mock_search_models.return_value = [{"id": "model1/test"}]
        mock_search_datasets.return_value = [{"id": "dataset1/test"}]
        mock_format.return_value = "Formatted results"

        result = search_huggingface_function("test query")
        assert result == "Formatted results"
        mock_search_models.assert_called_once_with("test query", limit=3)
        mock_search_datasets.assert_called_once_with("test query", limit=3)

    @patch("routers.search.utils.huggingface.search_models")
    @patch("routers.search.utils.huggingface.search_datasets")
    @patch("routers.search.utils.huggingface.format_search_results")
    def test_search_models_only(
        self, mock_format, mock_search_datasets, mock_search_models
    ):
        """Test searching for models only."""
        mock_search_models.return_value = [{"id": "model1/test"}]
        mock_format.return_value = "Formatted results"

        result = search_huggingface_function(
            "test query", include_models=True, include_datasets=False
        )
        assert result == "Formatted results"
        mock_search_models.assert_called_once()
        mock_search_datasets.assert_not_called()

    @patch("routers.search.utils.huggingface.search_models")
    @patch("routers.search.utils.huggingface.search_datasets")
    @patch("routers.search.utils.huggingface.format_search_results")
    def test_search_datasets_only(
        self, mock_format, mock_search_datasets, mock_search_models
    ):
        """Test searching for datasets only."""
        mock_search_datasets.return_value = [{"id": "dataset1/test"}]
        mock_format.return_value = "Formatted results"

        result = search_huggingface_function(
            "test query", include_models=False, include_datasets=True
        )
        assert result == "Formatted results"
        mock_search_models.assert_not_called()
        mock_search_datasets.assert_called_once()
