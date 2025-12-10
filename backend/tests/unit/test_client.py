"""Unit tests for client.py router functions."""

from unittest.mock import Mock
from routers.client import (
    _extract_model_ids_from_text,
    _extract_model_ids_from_graph,
    _serialize_graph_with_datasets,
)
from routers.search.utils.search_neo4j import (
    HFGraphData,
    HFNodes,
    HFModel,
    HFDataset,
    HFRelationships,
)


class TestExtractModelIdsFromText:
    """Tests for _extract_model_ids_from_text function."""

    def test_extract_from_markdown_list(self):
        """Test extraction from markdown formatted list."""
        text = "**1. [Qwen/Qwen3-4B]**\n**2. [bert-base-uncased]**"
        result = _extract_model_ids_from_text(text)
        # The pattern matches [author/model] so both should be found
        assert len(result) >= 1
        assert any("Qwen" in r or "bert" in r for r in result)

    def test_extract_from_bullet_list(self):
        """Test extraction from bullet list."""
        text = "- Qwen/Qwen3-4B\n- bert-base-uncased"
        result = _extract_model_ids_from_text(text)
        # The pattern matches - author/model-name
        assert len(result) >= 1
        assert any("Qwen" in r or "bert" in r for r in result)

    def test_extract_from_brackets(self):
        """Test extraction from bracket notation."""
        text = "Model [Qwen/Qwen3-4B] is available"
        result = _extract_model_ids_from_text(text)
        assert "Qwen/Qwen3-4B" in result

    def test_extract_from_word_boundary(self):
        """Test extraction using word boundaries."""
        text = "Check out Qwen/Qwen3-4B and allenai/c4"
        result = _extract_model_ids_from_text(text)
        assert "Qwen/Qwen3-4B" in result
        assert "allenai/c4" in result

    def test_limits_to_five(self):
        """Test that function limits results to 5."""
        text = "1. model1/test 2. model2/test 3. model3/test 4. model4/test 5. model5/test 6. model6/test"
        result = _extract_model_ids_from_text(text)
        assert len(result) == 5

    def test_deduplicates(self):
        """Test that duplicates are removed."""
        text = "Qwen/Qwen3-4B appears multiple times: Qwen/Qwen3-4B"
        result = _extract_model_ids_from_text(text)
        assert result.count("Qwen/Qwen3-4B") == 1

    def test_filters_urls(self):
        """Test that URLs are filtered out."""
        text = "Check https://huggingface.co/Qwen/Qwen3-4B"
        result = _extract_model_ids_from_text(text)
        assert "huggingface.co" not in str(result)

    def test_filters_http_urls(self):
        """Test that HTTP URLs are filtered out."""
        text = "Visit http://example.com/Qwen/Qwen3-4B"
        result = _extract_model_ids_from_text(text)
        assert len(result) == 0 or "http" not in str(result)

    def test_empty_text(self):
        """Test with empty text."""
        result = _extract_model_ids_from_text("")
        assert result == []

    def test_no_matches(self):
        """Test with text that has no matches."""
        result = _extract_model_ids_from_text("This is just regular text")
        assert result == []

    def test_invalid_format(self):
        """Test with invalid format (no slash)."""
        result = _extract_model_ids_from_text("This is not a valid model id")
        assert result == []


class TestExtractModelIdsFromGraph:
    """Tests for _extract_model_ids_from_graph function."""

    def test_extract_from_graph_with_model_nodes(self):
        """Test extraction from graph with model nodes."""
        # Create mock graph
        nodes = [
            HFModel(model_id="model1/test", downloads=1000),
            HFModel(model_id="model2/test", downloads=2000),
        ]
        graph = HFGraphData(
            nodes=HFNodes(nodes=nodes),
            relationships=HFRelationships(relationships=[]),
            queried_model_id="model1/test",
        )
        result = _extract_model_ids_from_graph(graph)
        assert "model1/test" in result
        assert "model2/test" in result

    def test_extract_from_graph_with_dataset_nodes(self):
        """Test extraction from graph with dataset nodes."""
        nodes = [
            HFDataset(dataset_id="dataset1/test"),
            HFDataset(dataset_id="dataset2/test"),
        ]
        graph = HFGraphData(
            nodes=HFNodes(nodes=nodes),
            relationships=HFRelationships(relationships=[]),
            queried_model_id="dataset1/test",
        )
        result = _extract_model_ids_from_graph(graph)
        assert "dataset1/test" in result
        assert "dataset2/test" in result

    def test_queried_model_id_at_front(self):
        """Test that queried_model_id is placed at front."""
        nodes = [HFModel(model_id="other/model", downloads=1000)]
        graph = HFGraphData(
            nodes=HFNodes(nodes=nodes),
            relationships=HFRelationships(relationships=[]),
            queried_model_id="queried/model",
        )
        result = _extract_model_ids_from_graph(graph)
        assert result[0] == "queried/model"

    def test_deduplicates_entity_ids(self):
        """Test that duplicate entity IDs are removed."""
        nodes = [
            HFModel(model_id="model1/test", downloads=1000),
            HFModel(model_id="model1/test", downloads=2000),
        ]
        graph = HFGraphData(
            nodes=HFNodes(nodes=nodes),
            relationships=HFRelationships(relationships=[]),
            queried_model_id="model1/test",
        )
        result = _extract_model_ids_from_graph(graph)
        assert result.count("model1/test") == 1

    def test_respects_limit(self):
        """Test that limit parameter is respected."""
        nodes = [HFModel(model_id=f"model{i}/test", downloads=1000) for i in range(15)]
        graph = HFGraphData(
            nodes=HFNodes(nodes=nodes),
            relationships=HFRelationships(relationships=[]),
            queried_model_id="queried/model",
        )
        result = _extract_model_ids_from_graph(graph, limit=5)
        assert len(result) == 5

    def test_empty_graph(self):
        """Test with empty graph."""
        graph = HFGraphData(
            nodes=HFNodes(nodes=[]),
            relationships=HFRelationships(relationships=[]),
            queried_model_id=None,
        )
        result = _extract_model_ids_from_graph(graph)
        assert result == []

    def test_none_graph(self):
        """Test with None graph."""
        result = _extract_model_ids_from_graph(None)
        assert result == []

    def test_dict_nodes(self):
        """Test with dict-based nodes."""
        graph = Mock()
        graph.nodes = Mock()
        graph.nodes.nodes = [
            {"model_id": "model1/test"},
            {"dataset_id": "dataset1/test"},
        ]
        graph.queried_model_id = "model1/test"
        result = _extract_model_ids_from_graph(graph)
        assert "model1/test" in result
        assert "dataset1/test" in result


class TestSerializeGraphWithDatasets:
    """Tests for _serialize_graph_with_datasets function."""

    def test_serialize_with_training_datasets(self):
        """Test serialization with training datasets."""
        nodes = [HFModel(model_id="model1/test", downloads=1000)]
        graph = HFGraphData(
            nodes=HFNodes(nodes=nodes),
            relationships=HFRelationships(relationships=[]),
            queried_model_id="model1/test",
        )
        training_datasets = {
            "model1/test": {
                "datasets": [{"name": "dataset1", "url": "http://example.com"}]
            }
        }
        result = _serialize_graph_with_datasets(graph, training_datasets)
        assert result is not None
        assert "nodes" in result
        nodes_list = result["nodes"]["nodes"]
        assert len(nodes_list) == 1
        assert nodes_list[0]["model_id"] == "model1/test"
        assert "training_datasets" in nodes_list[0]

    def test_serialize_without_training_datasets(self):
        """Test serialization without training datasets."""
        nodes = [HFModel(model_id="model1/test", downloads=1000)]
        graph = HFGraphData(
            nodes=HFNodes(nodes=nodes),
            relationships=HFRelationships(relationships=[]),
            queried_model_id="model1/test",
        )
        result = _serialize_graph_with_datasets(graph, None)
        assert result is not None
        assert "nodes" in result

    def test_serialize_empty_graph(self):
        """Test serialization of empty graph."""
        graph = HFGraphData(
            nodes=HFNodes(nodes=[]),
            relationships=HFRelationships(relationships=[]),
            queried_model_id=None,
        )
        result = _serialize_graph_with_datasets(graph, {})
        assert result is not None
        assert result["nodes"]["nodes"] == []

    def test_serialize_none_graph(self):
        """Test serialization with None graph."""
        result = _serialize_graph_with_datasets(None, {})
        assert result is None

    def test_non_dict_training_datasets(self):
        """Test with non-dict training_datasets."""
        nodes = [HFModel(model_id="model1/test", downloads=1000)]
        graph = HFGraphData(
            nodes=HFNodes(nodes=nodes),
            relationships=HFRelationships(relationships=[]),
            queried_model_id="model1/test",
        )
        result = _serialize_graph_with_datasets(graph, "not a dict")
        assert result is not None
        # Should still serialize graph even if training_datasets is invalid

    def test_matching_model_id(self):
        """Test that training datasets are matched by model_id."""
        nodes = [
            HFModel(model_id="model1/test", downloads=1000),
            HFModel(model_id="model2/test", downloads=2000),
        ]
        graph = HFGraphData(
            nodes=HFNodes(nodes=nodes),
            relationships=HFRelationships(relationships=[]),
            queried_model_id="model1/test",
        )
        training_datasets = {
            "model1/test": {"datasets": [{"name": "dataset1"}]},
            "model2/test": {"datasets": [{"name": "dataset2"}]},
        }
        result = _serialize_graph_with_datasets(graph, training_datasets)
        nodes_list = result["nodes"]["nodes"]
        model1_node = next(n for n in nodes_list if n["model_id"] == "model1/test")
        model2_node = next(n for n in nodes_list if n["model_id"] == "model2/test")
        assert "training_datasets" in model1_node
        assert "training_datasets" in model2_node
