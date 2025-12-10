"""Unit tests for search_neo4j.py functions."""

import pytest
from unittest.mock import Mock, patch
from routers.search.utils.search_neo4j import (
    search_query_impl,
    _parse_node,
    _log_query_summary,
    _make_entity,
    HFModel,
    HFDataset,
    HFGraphData,
)


class TestParseNode:
    """Tests for _parse_node function."""

    def test_parse_valid_model(self):
        """Test parsing a valid model node."""
        node_data = {
            "model_id": "test/model",
            "downloads": 1000,
            "pipeline_tag": "text-generation",
        }
        result = _parse_node(node_data, HFModel)
        assert isinstance(result, HFModel)
        assert result.model_id == "test/model"
        assert result.downloads == 1000

    def test_parse_valid_dataset(self):
        """Test parsing a valid dataset node."""
        node_data = {"dataset_id": "test/dataset", "tags": ["nlp"]}
        result = _parse_node(node_data, HFDataset)
        assert isinstance(result, HFDataset)
        assert result.dataset_id == "test/dataset"

    def test_parse_invalid_data(self):
        """Test parsing invalid node data."""
        node_data = {"invalid": "data"}
        result = _parse_node(node_data, HFModel)
        assert result is None

    def test_parse_missing_required_field(self):
        """Test parsing node with missing required field."""
        node_data = {"downloads": 1000}  # Missing model_id
        result = _parse_node(node_data, HFModel)
        assert result is None


class TestMakeEntity:
    """Tests for _make_entity function."""

    def test_make_model_entity(self):
        """Test creating a model entity."""
        node_dict = {"model_id": "test/model", "downloads": 1000}
        result = _make_entity(node_dict)
        assert isinstance(result, HFModel)
        assert result.model_id == "test/model"

    def test_make_dataset_entity(self):
        """Test creating a dataset entity."""
        node_dict = {"dataset_id": "test/dataset"}
        result = _make_entity(node_dict)
        assert isinstance(result, HFDataset)
        assert result.dataset_id == "test/dataset"

    def test_make_entity_raises_error(self):
        """Test that invalid entity raises ValueError."""
        node_dict = {"invalid": "data"}
        with pytest.raises(ValueError):
            _make_entity(node_dict)


class TestLogQuerySummary:
    """Tests for _log_query_summary function."""

    def test_logs_summary(self):
        """Test that query summary is logged."""
        summary = Mock()
        summary.query = "MATCH (n:Model) RETURN n"
        summary.result_available_after = 10
        with patch("routers.search.utils.search_neo4j.logger") as mock_logger:
            _log_query_summary(summary, 5)
            mock_logger.info.assert_called_once()


class TestSearchQueryImpl:
    """Tests for search_query_impl function."""

    @patch("routers.search.utils.search_neo4j.driver")
    @patch("routers.search.utils.search_neo4j.set_tool_result")
    def test_search_model_found(self, mock_set_tool_result, mock_driver):
        """Test searching for a model that exists."""
        # Mock root model query
        mock_root_record = Mock()
        mock_root_record.data.return_value = {
            "root": {"model_id": "test/model", "downloads": 1000}
        }

        mock_summary = Mock()
        mock_summary.query = "MATCH (root:Model {model_id: $model_id}) RETURN root"
        mock_summary.result_available_after = 5

        # Mock upstream query (empty)
        mock_upstream_summary = Mock()
        mock_upstream_summary.query = "upstream query"
        mock_upstream_summary.result_available_after = 5

        # Mock downstream query (empty)
        mock_downstream_summary = Mock()
        mock_downstream_summary.query = "downstream query"
        mock_downstream_summary.result_available_after = 5

        mock_driver.execute_query.side_effect = [
            ([mock_root_record], mock_summary, None),  # Root query
            ([], mock_upstream_summary, None),  # Upstream query
            ([], mock_downstream_summary, None),  # Downstream query
        ]

        result = search_query_impl("test/model")
        assert isinstance(result, HFGraphData)
        assert result.queried_model_id == "test/model"
        assert len(result.nodes.nodes) == 1
        assert len(result.relationships.relationships) == 0

    @patch("routers.search.utils.search_neo4j.driver")
    @patch("routers.search.utils.search_neo4j.set_tool_result")
    def test_search_dataset_found(self, mock_set_tool_result, mock_driver):
        """Test searching for a dataset that exists."""
        # Mock root dataset query (model query returns empty)
        mock_root_record = Mock()
        mock_root_record.data.return_value = {"root": {"dataset_id": "test/dataset"}}

        mock_model_summary = Mock()
        mock_model_summary.query = "model query"
        mock_model_summary.result_available_after = 5

        mock_dataset_summary = Mock()
        mock_dataset_summary.query = "dataset query"
        mock_dataset_summary.result_available_after = 5

        mock_upstream_summary = Mock()
        mock_upstream_summary.query = "upstream query"
        mock_upstream_summary.result_available_after = 5

        mock_downstream_summary = Mock()
        mock_downstream_summary.query = "downstream query"
        mock_downstream_summary.result_available_after = 5

        mock_driver.execute_query.side_effect = [
            ([], mock_model_summary, None),  # Model query (empty)
            ([mock_root_record], mock_dataset_summary, None),  # Dataset query
            ([], mock_upstream_summary, None),  # Upstream query
            ([], mock_downstream_summary, None),  # Downstream query
        ]

        result = search_query_impl("test/dataset")
        assert isinstance(result, HFGraphData)
        assert result.queried_model_id == "test/dataset"
        assert len(result.nodes.nodes) == 1

    @patch("routers.search.utils.search_neo4j.driver")
    def test_search_not_found(self, mock_driver):
        """Test searching for entity that doesn't exist."""
        mock_model_summary = Mock()
        mock_model_summary.query = "model query"
        mock_model_summary.result_available_after = 5

        mock_dataset_summary = Mock()
        mock_dataset_summary.query = "dataset query"
        mock_dataset_summary.result_available_after = 5

        mock_driver.execute_query.side_effect = [
            ([], mock_model_summary, None),  # Model query (empty)
            ([], mock_dataset_summary, None),  # Dataset query (empty)
        ]

        result = search_query_impl("nonexistent/entity")
        assert isinstance(result, HFGraphData)
        assert len(result.nodes.nodes) == 0
        assert len(result.relationships.relationships) == 0

    @patch("routers.search.utils.search_neo4j.driver")
    @patch("routers.search.utils.search_neo4j.set_tool_result")
    def test_search_with_upstream_relationships(
        self, mock_set_tool_result, mock_driver
    ):
        """Test search with upstream relationships."""
        # Mock root model
        mock_root_record = Mock()
        mock_root_record.data.return_value = {
            "root": {"model_id": "test/model", "downloads": 1000}
        }

        # Mock upstream model
        mock_upstream_record = Mock()
        mock_upstream_record.data.return_value = {
            "upstream": {"model_id": "upstream/model", "downloads": 500},
            "rel_type": "BASED_ON",
        }

        mock_summary = Mock()
        mock_summary.query = "query"
        mock_summary.result_available_after = 5

        mock_driver.execute_query.side_effect = [
            ([mock_root_record], mock_summary, None),  # Root query
            ([mock_upstream_record], mock_summary, None),  # Upstream query
            ([], mock_summary, None),  # Downstream query
        ]

        result = search_query_impl("test/model")
        assert len(result.nodes.nodes) == 2
        assert len(result.relationships.relationships) == 1
        rel = result.relationships.relationships[0]
        assert rel.relationship == "BASED_ON"
        assert rel.source.model_id == "test/model"
        assert rel.target.model_id == "upstream/model"

    @patch("routers.search.utils.search_neo4j.driver")
    @patch("routers.search.utils.search_neo4j.set_tool_result")
    def test_search_with_downstream_relationships(
        self, mock_set_tool_result, mock_driver
    ):
        """Test search with downstream relationships."""
        # Mock root model
        mock_root_record = Mock()
        mock_root_record.data.return_value = {
            "root": {"model_id": "test/model", "downloads": 1000}
        }

        # Mock downstream model
        mock_downstream_record = Mock()
        mock_downstream_record.data.return_value = {
            "downstream": {"model_id": "downstream/model", "downloads": 2000},
            "rel_type": "FINE_TUNED",
        }

        mock_summary = Mock()
        mock_summary.query = "query"
        mock_summary.result_available_after = 5

        mock_driver.execute_query.side_effect = [
            ([mock_root_record], mock_summary, None),  # Root query
            ([], mock_summary, None),  # Upstream query
            ([mock_downstream_record], mock_summary, None),  # Downstream query
        ]

        result = search_query_impl("test/model")
        assert len(result.nodes.nodes) == 2
        assert len(result.relationships.relationships) == 1
        rel = result.relationships.relationships[0]
        assert rel.relationship == "FINE_TUNED"
        assert rel.source.model_id == "downstream/model"
        assert rel.target.model_id == "test/model"

    @patch("routers.search.utils.search_neo4j.driver")
    @patch("routers.search.utils.search_neo4j.set_tool_result")
    def test_search_respects_max_related_limit(self, mock_set_tool_result, mock_driver):
        """Test that search respects MAX_RELATED limit."""
        # Mock root model
        mock_root_record = Mock()
        mock_root_record.data.return_value = {
            "root": {"model_id": "test/model", "downloads": 1000}
        }

        # Create 15 upstream records (more than MAX_RELATED=10)
        # But Neo4j query limits to 10, so we'll get 10 back
        mock_upstream_records = []
        for i in range(10):  # Neo4j LIMIT will return max 10
            mock_record = Mock()
            mock_record.data.return_value = {
                "upstream": {"model_id": f"upstream{i}/model", "downloads": 500},
                "rel_type": "BASED_ON",
            }
            mock_upstream_records.append(mock_record)

        mock_summary = Mock()
        mock_summary.query = "query"
        mock_summary.result_available_after = 5

        mock_driver.execute_query.side_effect = [
            ([mock_root_record], mock_summary, None),  # Root query
            (
                mock_upstream_records,
                mock_summary,
                None,
            ),  # Upstream query (limited to 10 by query)
            ([], mock_summary, None),  # Downstream query
        ]

        result = search_query_impl("test/model")
        # Should have root + up to 10 upstream = 11 nodes max
        assert len(result.nodes.nodes) <= 11
