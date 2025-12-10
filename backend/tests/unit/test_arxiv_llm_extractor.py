"""Unit tests for arxiv_llm_extractor.py functions."""

import json
from unittest.mock import Mock, patch
from routers.search.utils.arxiv_llm_extractor import LLMDatasetExtractor


class TestLLMDatasetExtractor:
    """Tests for LLMDatasetExtractor class."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch(
                "routers.search.utils.arxiv_llm_extractor.OpenAI"
            ) as mock_openai:
                extractor = LLMDatasetExtractor()
                assert extractor.client is not None
                mock_openai.assert_called_once_with(api_key="test-key")

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        with patch.dict("os.environ", {}, clear=True):
            extractor = LLMDatasetExtractor()
            assert extractor.client is None

    def test_is_available_with_client(self):
        """Test is_available when client exists."""
        extractor = LLMDatasetExtractor()
        extractor.client = Mock()
        assert extractor.is_available() is True

    def test_is_available_without_client(self):
        """Test is_available when client doesn't exist."""
        extractor = LLMDatasetExtractor()
        extractor.client = None
        assert extractor.is_available() is False

    @patch("routers.search.utils.arxiv_llm_extractor.OpenAI")
    def test_extract_datasets_success(self, mock_openai_class):
        """Test successful dataset extraction."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "datasets": [
                    {
                        "name": "BookCorpus",
                        "type": "public_dataset",
                        "source": None,
                        "context": "Used for pretraining",
                        "hf_url": "https://huggingface.co/datasets/bookcorpus",
                    },
                    {
                        "name": "Synthetic data from GPT-4",
                        "type": "synthetic",
                        "source": "GPT-4",
                        "context": "Generated for training",
                        "hf_url": None,
                    },
                ]
            }
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        extractor = LLMDatasetExtractor()
        extractor.client = mock_client

        result = extractor.extract_datasets(
            "Paper text about training", "model/test", "https://arxiv.org/abs/1234.5678"
        )

        assert len(result) == 2
        assert result[0].name == "BookCorpus"
        assert result[0].type == "public_dataset"
        assert result[1].name == "Synthetic data from GPT-4"
        assert result[1].type == "synthetic"

    @patch("routers.search.utils.arxiv_llm_extractor.OpenAI")
    def test_extract_datasets_no_client(self, mock_openai_class):
        """Test extraction when client is not available."""
        extractor = LLMDatasetExtractor()
        extractor.client = None

        result = extractor.extract_datasets(
            "Paper text", "model/test", "https://arxiv.org/abs/1234.5678"
        )

        assert result == []

    @patch("routers.search.utils.arxiv_llm_extractor.OpenAI")
    def test_extract_datasets_empty_response(self, mock_openai_class):
        """Test extraction with empty response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        extractor = LLMDatasetExtractor()
        extractor.client = mock_client

        result = extractor.extract_datasets(
            "Paper text", "model/test", "https://arxiv.org/abs/1234.5678"
        )

        assert result == []

    @patch("routers.search.utils.arxiv_llm_extractor.OpenAI")
    def test_extract_datasets_json_decode_error(self, mock_openai_class):
        """Test extraction with JSON decode error."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid JSON"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        extractor = LLMDatasetExtractor()
        extractor.client = mock_client

        with patch("routers.search.utils.arxiv_llm_extractor.logger") as mock_logger:
            result = extractor.extract_datasets(
                "Paper text", "model/test", "https://arxiv.org/abs/1234.5678"
            )
            assert result == []
            mock_logger.error.assert_called()

    @patch("routers.search.utils.arxiv_llm_extractor.OpenAI")
    def test_extract_datasets_api_exception(self, mock_openai_class):
        """Test extraction with API exception."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai_class.return_value = mock_client

        extractor = LLMDatasetExtractor()
        extractor.client = mock_client

        with patch("routers.search.utils.arxiv_llm_extractor.logger") as mock_logger:
            result = extractor.extract_datasets(
                "Paper text", "model/test", "https://arxiv.org/abs/1234.5678"
            )
            assert result == []
            mock_logger.error.assert_called()

    @patch("routers.search.utils.arxiv_llm_extractor.OpenAI")
    def test_extract_datasets_limits_text(self, mock_openai_class):
        """Test that paper text is limited to avoid token limits."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({"datasets": []})
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        extractor = LLMDatasetExtractor()
        extractor.client = mock_client

        # Create very long text
        long_text = "A" * 20000
        extractor.extract_datasets(
            long_text, "model/test", "https://arxiv.org/abs/1234.5678"
        )

        # Check that the prompt was limited
        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][1]["content"]
        # Should be limited to roughly 10000 chars
        assert len(prompt) < len(long_text)

    @patch("routers.search.utils.arxiv_llm_extractor.OpenAI")
    def test_extract_datasets_invalid_dataset_entry(self, mock_openai_class):
        """Test extraction with invalid dataset entry in response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "datasets": [
                    {
                        "name": "Valid Dataset",
                        "type": "public_dataset",
                        "source": None,
                        "context": "Used for training",
                        "hf_url": None,
                    },
                    {
                        "invalid": "entry",  # Missing required fields
                    },
                ]
            }
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        extractor = LLMDatasetExtractor()
        extractor.client = mock_client

        with patch("routers.search.utils.arxiv_llm_extractor.logger"):
            result = extractor.extract_datasets(
                "Paper text", "model/test", "https://arxiv.org/abs/1234.5678"
            )
            # Should still return valid entries (invalid ones are skipped with warning)
            assert len(result) >= 1
            assert any(ds.name == "Valid Dataset" for ds in result)
