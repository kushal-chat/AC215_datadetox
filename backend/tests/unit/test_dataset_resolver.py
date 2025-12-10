"""Unit tests for dataset_resolver.py functions."""

from unittest.mock import Mock, patch
from huggingface_hub.utils import HfHubHTTPError
from routers.search.utils.dataset_resolver import (
    check_dataset_exists,
    resolve_dataset_url,
    enrich_dataset_info,
    _looks_like_dataset_id,
)


class TestLooksLikeDatasetId:
    """Tests for _looks_like_dataset_id function."""

    def test_valid_dataset_id(self):
        """Test valid dataset ID format."""
        assert _looks_like_dataset_id("squad")
        assert _looks_like_dataset_id("allenai/c4")
        assert _looks_like_dataset_id("model-name")
        assert _looks_like_dataset_id("author/model-name")

    def test_invalid_dataset_id(self):
        """Test invalid dataset ID format."""
        assert not _looks_like_dataset_id("dataset with spaces")
        assert not _looks_like_dataset_id("dataset/with/slashes")
        assert not _looks_like_dataset_id("")
        assert not _looks_like_dataset_id("dataset@invalid")


class TestCheckDatasetExists:
    """Tests for check_dataset_exists function."""

    @patch("routers.search.utils.dataset_resolver.hf_api")
    def test_check_dataset_exists_true(self, mock_api):
        """Test checking existing dataset."""
        mock_api.dataset_info.return_value = Mock()
        result = check_dataset_exists("squad")
        assert result is True
        mock_api.dataset_info.assert_called_once_with("squad")

    @patch("routers.search.utils.dataset_resolver.hf_api")
    def test_check_dataset_exists_404(self, mock_api):
        """Test checking non-existent dataset (404)."""
        mock_error = HfHubHTTPError("Not found")
        mock_error.response = Mock()
        mock_error.response.status_code = 404
        mock_api.dataset_info.side_effect = mock_error

        result = check_dataset_exists("nonexistent/dataset")
        assert result is False

    @patch("routers.search.utils.dataset_resolver.hf_api")
    def test_check_dataset_exists_other_error(self, mock_api):
        """Test checking dataset with other HTTP error."""
        mock_error = HfHubHTTPError("Server error")
        mock_error.response = Mock()
        mock_error.response.status_code = 500
        mock_api.dataset_info.side_effect = mock_error

        with patch("routers.search.utils.dataset_resolver.logger") as mock_logger:
            result = check_dataset_exists("dataset/test")
            assert result is False
            mock_logger.warning.assert_called_once()

    @patch("routers.search.utils.dataset_resolver.hf_api")
    def test_check_dataset_exists_general_exception(self, mock_api):
        """Test checking dataset with general exception."""
        mock_api.dataset_info.side_effect = Exception("Unexpected error")

        with patch("routers.search.utils.dataset_resolver.logger") as mock_logger:
            result = check_dataset_exists("dataset/test")
            assert result is False
            mock_logger.warning.assert_called_once()

    def test_check_dataset_exists_invalid_format(self):
        """Test checking dataset with invalid format."""
        result = check_dataset_exists("dataset with spaces")
        assert result is False

    @patch("routers.search.utils.dataset_resolver.hf_api")
    def test_check_dataset_exists_caching(self, mock_api):
        """Test that dataset existence is cached."""
        mock_api.dataset_info.return_value = Mock()

        # First call
        result1 = check_dataset_exists("cached/dataset")
        assert result1 is True

        # Second call should use cache
        result2 = check_dataset_exists("cached/dataset")
        assert result2 is True
        # Should only be called once due to caching
        assert mock_api.dataset_info.call_count == 1


class TestResolveDatasetUrl:
    """Tests for resolve_dataset_url function."""

    def test_resolve_with_existing_url(self):
        """Test resolving with existing URL."""
        result = resolve_dataset_url("squad", "https://huggingface.co/datasets/squad")
        assert result == "https://huggingface.co/datasets/squad"

    def test_resolve_without_existing_url(self):
        """Test resolving without existing URL."""
        result = resolve_dataset_url("squad", None)
        assert result == "https://huggingface.co/datasets?search=squad"

    def test_resolve_lowercase(self):
        """Test that dataset name is lowercased in search URL."""
        result = resolve_dataset_url("SQUAD", None)
        assert "squad" in result.lower()

    def test_resolve_strips_whitespace(self):
        """Test that whitespace is stripped."""
        result = resolve_dataset_url("  squad  ", None)
        assert "squad" in result


class TestEnrichDatasetInfo:
    """Tests for enrich_dataset_info function."""

    def test_enrich_with_existing_url(self):
        """Test enriching with existing URL."""
        datasets = [
            {
                "name": "squad",
                "url": "https://huggingface.co/datasets/squad",
                "description": "Stanford dataset",
            }
        ]
        result = enrich_dataset_info(datasets)
        assert len(result) == 1
        assert result[0]["url"] == "https://huggingface.co/datasets/squad"

    def test_enrich_without_url(self):
        """Test enriching without URL."""
        datasets = [
            {
                "name": "squad",
                "url": None,
                "description": "Stanford dataset",
            }
        ]
        result = enrich_dataset_info(datasets)
        assert len(result) == 1
        assert result[0]["url"] is not None
        assert "huggingface.co/datasets?search=" in result[0]["url"]

    def test_enrich_multiple_datasets(self):
        """Test enriching multiple datasets."""
        datasets = [
            {"name": "squad", "url": None, "description": "Dataset 1"},
            {"name": "glue", "url": None, "description": "Dataset 2"},
        ]
        result = enrich_dataset_info(datasets)
        assert len(result) == 2
        assert all("url" in ds for ds in result)

    def test_enrich_preserves_description(self):
        """Test that description is preserved."""
        datasets = [
            {
                "name": "squad",
                "url": None,
                "description": "Stanford Question Answering Dataset",
            }
        ]
        result = enrich_dataset_info(datasets)
        assert result[0]["description"] == "Stanford Question Answering Dataset"

    def test_enrich_with_name_only(self):
        """Test enriching with only name."""
        datasets = [{"name": "squad"}]
        result = enrich_dataset_info(datasets)
        assert len(result) == 1
        assert result[0]["name"] == "squad"
        assert "url" in result[0]

    def test_enrich_empty_list(self):
        """Test enriching empty list."""
        result = enrich_dataset_info([])
        assert result == []
