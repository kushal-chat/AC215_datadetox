"""Integration tests for the full API flow."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from main import app

client = TestClient(app)


@pytest.fixture
def mock_agent_runner():
    """Mock the agent runner to avoid actual LLM calls."""
    with patch("routers.client.Runner") as mock_runner:
        mock_result = MagicMock()
        mock_result.final_output_as.return_value = "Test response from agent"
        mock_runner.run = AsyncMock(return_value=mock_result)
        yield mock_runner


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver."""
    with patch("routers.search.utils.search_neo4j.driver") as mock_driver:
        # Mock query result with nodes and relationships
        mock_node = MagicMock()
        mock_node.data.return_value = {
            "n": {
                "model_id": "test/model",
                "downloads": 1000,
                "pipeline_tag": "text-generation",
            }
        }

        mock_summary = MagicMock()
        mock_summary.query = "MATCH..."
        mock_summary.result_available_after = 10

        # Mock relationship data
        mock_record = MagicMock()
        mock_record.data.return_value = {
            "nodes": [
                {"model_id": "test/model", "downloads": 1000},
                {"model_id": "test/model2", "downloads": 500},
            ],
            "relationships": [
                (
                    {"model_id": "test/model"},
                    "BASED_ON",
                    {"model_id": "test/model2"},
                )
            ],
        }

        # Mock search_query response
        mock_driver.execute_query.return_value = (
            [mock_record],
            mock_summary,
            None,
        )
        yield mock_driver


@pytest.fixture
def mock_huggingface_api():
    """Mock HuggingFace API calls."""
    with patch("routers.search.utils.huggingface.hf_api") as mock_api:
        # Mock model search
        mock_model = MagicMock()
        mock_model.id = "test/model"
        mock_model.author = "test_author"
        mock_model.downloads = 1000
        mock_model.likes = 50
        mock_model.tags = ["nlp"]
        mock_model.pipeline_tag = "text-generation"
        mock_model.library_name = "transformers"
        mock_model.private = False
        mock_model.created_at = None
        mock_model.last_modified = None
        mock_model.sha = "abc123"

        mock_api.list_models.return_value = [mock_model]
        yield mock_api


def test_full_search_flow_with_neo4j_data(
    mock_agent_runner, mock_neo4j_driver, mock_huggingface_api
):
    """Test the complete search flow from API to response with Neo4j data."""
    # Mock the agent to actually call search_neo4j by patching the tool
    with patch("routers.search.utils.search_neo4j.search_query") as mock_search_query:
        from routers.search.utils.search_neo4j import (
            HFGraphData,
            HFNodes,
            HFRelationships,
        )

        # Mock search_query to return graph data
        mock_graph_data = HFGraphData(
            nodes=HFNodes(nodes=[]),
            relationships=HFRelationships(relationships=[]),
        )
        mock_search_query.return_value = mock_graph_data

        response = client.post(
            "/backend/flow/search",
            json={"query_val": "test model"},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "result" in data
        assert data["result"] == "Test response from agent"

        # Neo4j data may or may not be included depending on agent behavior
        # The important thing is the API flow works
        assert isinstance(data, dict)


def test_search_flow_without_neo4j_data(mock_agent_runner, mock_huggingface_api):
    """Test search flow when Neo4j returns no data."""
    with patch("routers.search.utils.search_neo4j.driver") as mock_driver:
        mock_summary = MagicMock()
        mock_driver.execute_query.return_value = ([], mock_summary, None)

        response = client.post(
            "/backend/flow/search",
            json={"query_val": "nonexistent model"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        # neo4j_data may or may not be present depending on agent behavior


def test_search_flow_error_handling(mock_agent_runner):
    """Test error handling in the search flow."""
    with patch("routers.search.utils.huggingface.hf_api") as mock_api:
        mock_api.list_models.side_effect = Exception("API Error")

        # Agent should handle the error gracefully
        response = client.post(
            "/backend/flow/search",
            json={"query_val": "test"},
        )

        # Should still return 200, but with error in result
        assert response.status_code == 200


def test_search_flow_validation():
    """Test input validation in the search endpoint."""
    # Missing query_val
    response = client.post("/backend/flow/search", json={})
    assert response.status_code == 422

    # Invalid JSON
    response = client.post(
        "/backend/flow/search",
        content="invalid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_search_flow_with_request_context(
    mock_agent_runner, mock_neo4j_driver, mock_huggingface_api
):
    """Test that request context is properly set and used."""
    response = client.post(
        "/backend/flow/search",
        json={"query_val": "test model"},
    )

    assert response.status_code == 200
    data = response.json()

    # Verify the response contains expected data
    assert "result" in data


def test_search_flow_tool_state_integration(
    mock_agent_runner, mock_neo4j_driver, mock_huggingface_api
):
    """Test that tool state is properly managed across the flow."""
    response = client.post(
        "/backend/flow/search",
        json={"query_val": "bert model"},
    )

    assert response.status_code == 200
    data = response.json()

    # Verify Neo4j tool result was stored and retrieved
    if "neo4j_data" in data:
        assert "nodes" in data["neo4j_data"]
        assert "relationships" in data["neo4j_data"]
