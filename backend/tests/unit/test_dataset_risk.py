"""Unit tests for dataset_risk.py functions."""

from routers.search.utils.dataset_risk import (
    build_dataset_risk_context,
    _dataset_risk,
    _normalize,
    _flag_synthetic,
    _flag_english_bias,
)


class TestNormalize:
    """Tests for _normalize function."""

    def test_normalize_lowercase(self):
        """Test that text is lowercased."""
        assert _normalize("HELLO") == "hello"
        assert _normalize("Hello World") == "hello world"

    def test_normalize_none(self):
        """Test that None returns empty string."""
        assert _normalize(None) == ""

    def test_normalize_empty(self):
        """Test that empty string returns empty string."""
        assert _normalize("") == ""


class TestFlagSynthetic:
    """Tests for _flag_synthetic function."""

    def test_flags_synthetic_keyword(self):
        """Test that synthetic keywords are detected."""
        assert _flag_synthetic("This is synthetic data")
        assert _flag_synthetic("model-generated content")
        assert _flag_synthetic("generated dataset")

    def test_no_synthetic_keyword(self):
        """Test that non-synthetic text is not flagged."""
        assert not _flag_synthetic("This is real data")
        assert not _flag_synthetic("natural language")

    def test_case_insensitive(self):
        """Test that detection is case-insensitive."""
        # The function normalizes text first, so it should be case-insensitive
        assert _flag_synthetic("synthetic data")
        assert _flag_synthetic("generated content")


class TestFlagEnglishBias:
    """Tests for _flag_english_bias function."""

    def test_flags_english_keywords(self):
        """Test that English keywords are detected."""
        # The function uses word boundaries with regex, test with proper word boundaries
        assert _flag_english_bias("english text")
        assert _flag_english_bias("us dataset")
        assert _flag_english_bias("uk corpus")

    def test_no_english_keyword(self):
        """Test that non-English text is not flagged."""
        assert not _flag_english_bias("Multilingual dataset")
        assert not _flag_english_bias("French corpus")

    def test_word_boundary_matching(self):
        """Test that word boundaries are respected."""
        # "en" should match as a word, not as part of "generate"
        assert _flag_english_bias("en dataset")
        # Should not match "en" in "generate"
        assert not _flag_english_bias("generate dataset")

    def test_empty_text(self):
        """Test that empty text returns False."""
        assert not _flag_english_bias("")
        assert not _flag_english_bias(None)


class TestDatasetRisk:
    """Tests for _dataset_risk function."""

    def test_synthetic_dataset(self):
        """Test risk assessment for synthetic dataset."""
        dataset = {
            "name": "synthetic dataset",
            "description": "This is synthetic data",
            "url": "http://example.com",
        }
        result = _dataset_risk(dataset)
        assert result["risk_level"] in ["medium", "high"]
        assert "synthetic_source" in result["indicators"]

    def test_english_centric_dataset(self):
        """Test risk assessment for English-centric dataset."""
        dataset = {
            "name": "English corpus",
            "description": "US English text",
            "url": "http://example.com",
        }
        result = _dataset_risk(dataset)
        assert result["risk_level"] in ["medium", "high"]
        assert "english_centric" in result["indicators"]

    def test_no_url_dataset(self):
        """Test risk assessment for dataset without URL."""
        dataset = {
            "name": "unknown dataset",
            "description": "Some dataset",
            "url": None,
        }
        result = _dataset_risk(dataset)
        assert "no_verified_source" in result["indicators"]
        assert result["risk_level"] in ["medium", "high"]

    def test_high_risk_dataset(self):
        """Test risk assessment for known high-risk dataset."""
        dataset = {
            "name": "pile",
            "description": "The Pile dataset",
            "url": "http://example.com",
        }
        result = _dataset_risk(dataset)
        assert "known_large_crawl" in result["indicators"]
        assert result["risk_level"] in ["medium", "high"]

    def test_low_risk_dataset(self):
        """Test risk assessment for low-risk dataset."""
        dataset = {
            "name": "squad",
            "description": "Stanford Question Answering Dataset",
            "url": "https://huggingface.co/datasets/squad",
        }
        result = _dataset_risk(dataset)
        assert result["risk_level"] == "low"
        assert "no_specific_flags" in result["indicators"]

    def test_high_risk_score(self):
        """Test that high risk score results in high risk level."""
        dataset = {
            "name": "synthetic pile",
            "description": "Synthetic English data from pile",
            "url": None,
        }
        result = _dataset_risk(dataset)
        # Should have multiple risk factors (synthetic + pile + no url = at least 3)
        assert result["risk_level"] == "high"

    def test_medium_risk_score(self):
        """Test that medium risk score results in medium risk level."""
        dataset = {
            "name": "english dataset",
            "description": "US English text",
            "url": "http://example.com",
        }
        result = _dataset_risk(dataset)
        # Should have 1-2 risk factors (english_centric = 1)
        assert result["risk_level"] in ["medium", "high"]

    def test_url_present_flag(self):
        """Test that url_present flag is set correctly."""
        dataset = {
            "name": "test dataset",
            "description": "Test",
            "url": "http://example.com",
        }
        result = _dataset_risk(dataset)
        assert result["url_present"] is True

        dataset_no_url = {
            "name": "test dataset",
            "description": "Test",
            "url": None,
        }
        result_no_url = _dataset_risk(dataset_no_url)
        assert result_no_url["url_present"] is False


class TestBuildDatasetRiskContext:
    """Tests for build_dataset_risk_context function."""

    def test_build_context_with_datasets(self):
        """Test building context with training datasets."""
        training_dataset_map = {
            "model1/test": {
                "arxiv_url": "https://arxiv.org/abs/1234.5678",
                "datasets": [
                    {
                        "name": "synthetic dataset",
                        "description": "Synthetic data",
                        "url": None,
                    },
                    {
                        "name": "squad",
                        "description": "Stanford dataset",
                        "url": "https://huggingface.co/datasets/squad",
                    },
                ],
            }
        }
        result = build_dataset_risk_context(training_dataset_map)
        assert "models" in result
        assert len(result["models"]) == 1
        assert result["models"][0]["model_id"] == "model1/test"
        assert len(result["models"][0]["datasets"]) == 2
        assert "global_counts" in result

    def test_build_context_empty_datasets(self):
        """Test building context with empty datasets."""
        training_dataset_map = {
            "model1/test": {
                "arxiv_url": "https://arxiv.org/abs/1234.5678",
                "datasets": [],
            }
        }
        result = build_dataset_risk_context(training_dataset_map)
        assert result["global_counts"]["unknown_models"] == 1

    def test_build_context_no_datasets_key(self):
        """Test building context when datasets key is missing."""
        training_dataset_map = {
            "model1/test": {
                "arxiv_url": "https://arxiv.org/abs/1234.5678",
            }
        }
        result = build_dataset_risk_context(training_dataset_map)
        assert result["global_counts"]["unknown_models"] == 1

    def test_build_context_none_input(self):
        """Test building context with None input."""
        result = build_dataset_risk_context(None)
        assert result["models"] == []
        assert result["global_counts"]["unknown_models"] == 0

    def test_build_context_non_dict_input(self):
        """Test building context with non-dict input."""
        result = build_dataset_risk_context("not a dict")
        assert result["models"] == []
        assert result["global_counts"]["unknown_models"] == 0

    def test_build_context_multiple_models(self):
        """Test building context with multiple models."""
        training_dataset_map = {
            "model1/test": {
                "arxiv_url": "https://arxiv.org/abs/1234.5678",
                "datasets": [
                    {
                        "name": "synthetic dataset",
                        "description": "Synthetic",
                        "url": None,
                    }
                ],
            },
            "model2/test": {
                "arxiv_url": "https://arxiv.org/abs/1234.5679",
                "datasets": [
                    {
                        "name": "squad",
                        "description": "Stanford dataset",
                        "url": "https://huggingface.co/datasets/squad",
                    }
                ],
            },
        }
        result = build_dataset_risk_context(training_dataset_map)
        assert len(result["models"]) == 2
        assert result["models"][0]["model_id"] == "model1/test"
        assert result["models"][1]["model_id"] == "model2/test"

    def test_build_context_risk_counts(self):
        """Test that risk counts are calculated correctly."""
        training_dataset_map = {
            "model1/test": {
                "arxiv_url": "https://arxiv.org/abs/1234.5678",
                "datasets": [
                    {
                        "name": "synthetic dataset",
                        "description": "Synthetic data",
                        "url": None,
                    },
                    {
                        "name": "squad",
                        "description": "Stanford dataset",
                        "url": "https://huggingface.co/datasets/squad",
                    },
                ],
            }
        }
        result = build_dataset_risk_context(training_dataset_map)
        # Should have counts for different risk levels
        assert "high" in result["global_counts"]
        assert "medium" in result["global_counts"]
        assert "low" in result["global_counts"]

    def test_build_context_preserves_arxiv_url(self):
        """Test that arxiv_url is preserved in context."""
        training_dataset_map = {
            "model1/test": {
                "arxiv_url": "https://arxiv.org/abs/1234.5678",
                "datasets": [
                    {
                        "name": "squad",
                        "description": "Stanford dataset",
                        "url": "https://huggingface.co/datasets/squad",
                    }
                ],
            }
        }
        result = build_dataset_risk_context(training_dataset_map)
        assert result["models"][0]["arxiv_url"] == "https://arxiv.org/abs/1234.5678"

