"""Integration tests for Neo4j search functionality with mocked database."""

import pytest
from unittest.mock import Mock, patch
from routers.search.utils.search_neo4j import (
    search_query_impl,
    HFGraphData,
)


@pytest.fixture(scope="function")
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    with patch("routers.search.utils.search_neo4j.driver") as mock_driver:
        yield mock_driver


@pytest.mark.integration
def test_neo4j_search_integration(mock_neo4j_driver):
    """Test searching Neo4j with mocked database."""
    # Mock the root model query
    mock_root_record = Mock()
    mock_root_record.data.return_value = {
        "root": {
            "model_id": "test/model",
            "downloads": 1000,
            "pipeline_tag": "text-generation",
            "library_name": "transformers",
        }
    }

    mock_summary = Mock()
    mock_summary.query = "MATCH (root:Model {model_id: $model_id}) RETURN root"
    mock_summary.result_available_after = 5

    # Mock empty upstream and downstream queries
    mock_empty_summary = Mock()
    mock_empty_summary.query = "query"
    mock_empty_summary.result_available_after = 5

    mock_neo4j_driver.execute_query.side_effect = [
        ([mock_root_record], mock_summary, None),  # Root query
        ([], mock_empty_summary, None),  # Upstream query
        ([], mock_empty_summary, None),  # Downstream query
    ]

    # Test search_query_impl
    result = search_query_impl("test/model")

    assert isinstance(result, HFGraphData)
    assert result.queried_model_id == "test/model"
    assert len(result.nodes.nodes) == 1
    assert result.nodes.nodes[0].model_id == "test/model"
    assert result.nodes.nodes[0].downloads == 1000


@pytest.mark.integration
def test_neo4j_relationships_integration(mock_neo4j_driver):
    """Test Neo4j relationships with mocked database."""
    # Mock root model
    mock_root_record = Mock()
    mock_root_record.data.return_value = {
        "root": {"model_id": "model1", "downloads": 1000}
    }

    # Mock upstream relationship
    mock_upstream_record = Mock()
    mock_upstream_record.data.return_value = {
        "upstream": {"model_id": "model2", "downloads": 500},
        "rel_type": "BASED_ON",
    }

    mock_summary = Mock()
    mock_summary.query = "query"
    mock_summary.result_available_after = 5

    mock_neo4j_driver.execute_query.side_effect = [
        ([mock_root_record], mock_summary, None),  # Root query
        ([mock_upstream_record], mock_summary, None),  # Upstream query
        ([], mock_summary, None),  # Downstream query
    ]

    # Test search_query_impl with relationships
    result = search_query_impl("model1")

    assert isinstance(result, HFGraphData)
    assert len(result.nodes.nodes) == 2
    assert len(result.relationships.relationships) == 1

    rel = result.relationships.relationships[0]
    assert rel.relationship == "BASED_ON"
    assert rel.source.model_id == "model1"
    assert rel.target.model_id == "model2"


@pytest.mark.integration
def test_neo4j_apoc_integration(mock_neo4j_driver):
    """Test Neo4j graph queries with multiple relationships."""
    # Mock root model
    mock_root_record = Mock()
    mock_root_record.data.return_value = {
        "root": {"model_id": "root", "downloads": 1000}
    }

    # Mock multiple upstream relationships
    mock_upstream1 = Mock()
    mock_upstream1.data.return_value = {
        "upstream": {"model_id": "child1", "downloads": 500},
        "rel_type": "BASED_ON",
    }

    mock_upstream2 = Mock()
    mock_upstream2.data.return_value = {
        "upstream": {"model_id": "child2", "downloads": 300},
        "rel_type": "BASED_ON",
    }

    mock_summary = Mock()
    mock_summary.query = "query"
    mock_summary.result_available_after = 5

    mock_neo4j_driver.execute_query.side_effect = [
        ([mock_root_record], mock_summary, None),  # Root query
        (
            [mock_upstream1, mock_upstream2],
            mock_summary,
            None,
        ),  # Upstream query (2 relationships)
        ([], mock_summary, None),  # Downstream query
    ]

    # Test search_query_impl with multiple relationships
    result = search_query_impl("root")

    assert isinstance(result, HFGraphData)
    assert len(result.nodes.nodes) == 3  # root + 2 children
    assert len(result.relationships.relationships) == 2  # 2 BASED_ON relationships

    # Verify all relationships are BASED_ON
    for rel in result.relationships.relationships:
        assert rel.relationship == "BASED_ON"
        assert rel.source.model_id == "root"
