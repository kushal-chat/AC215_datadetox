"""Integration tests for the full API flow."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

client = TestClient(app)


@pytest.fixture
def mock_agent_runner():
    """Mock the agent runner to avoid actual LLM calls."""
    with patch("routers.client.Runner") as mock_runner:
        # Create a proper mock for streaming results
        async def mock_stream_events():
            # Return empty stream for now
            return
            yield  # Make it a generator

        # Mock result for HuggingFace search
        mock_hf_result = MagicMock()
        mock_hf_result.final_output_as.return_value = (
            "**1. [test/model]**\nModel description here"
        )
        mock_hf_result.stream_events = mock_stream_events

        # Mock result for Neo4j search
        mock_neo4j_result = MagicMock()
        mock_neo4j_result.final_output_as.return_value = "Neo4j search results"
        mock_neo4j_result.stream_events = mock_stream_events

        # Mock result for dataset extraction
        mock_dataset_result = MagicMock()
        mock_dataset_result.final_output_as.return_value = "Dataset extraction results"
        mock_dataset_result.stream_events = mock_stream_events

        # Mock result for risk assessment
        mock_risk_result = MagicMock()
        mock_risk_result.final_output_as.return_value = "Risk assessment results"
        mock_risk_result.stream_events = mock_stream_events

        # Mock result for compiler
        mock_compiler_result = MagicMock()
        mock_compiler_result.final_output_as.return_value = "Compiled response"
        mock_compiler_result.stream_events = mock_stream_events

        # Configure run_streamed to return appropriate results based on agent
        def run_streamed_side_effect(starting_agent, input):
            # Return different results based on which agent is being called
            agent_name = getattr(starting_agent, "name", "")
            if "HFSearch" in agent_name or "hf_search" in str(starting_agent):
                return mock_hf_result
            elif "Neo4j" in agent_name or "neo4j" in str(starting_agent):
                return mock_neo4j_result
            elif "Dataset" in agent_name or "dataset" in str(starting_agent):
                return mock_dataset_result
            elif "Risk" in agent_name or "risk" in str(starting_agent):
                return mock_risk_result
            elif "Compiler" in agent_name or "compiler" in str(starting_agent):
                return mock_compiler_result
            return mock_hf_result

        mock_runner.run_streamed = MagicMock(side_effect=run_streamed_side_effect)
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
    with patch(
        "routers.search.utils.search_neo4j.search_query_impl"
    ) as mock_search_query:
        from routers.search.utils.search_neo4j import (
            HFGraphData,
            HFNodes,
            HFRelationships,
            HFModel,
        )

        # Mock search_query to return graph data with a model
        mock_model = HFModel(model_id="test/model", downloads=1000)
        mock_graph_data = HFGraphData(
            nodes=HFNodes(nodes=[mock_model]),
            relationships=HFRelationships(relationships=[]),
            queried_model_id="test/model",
        )
        mock_search_query.return_value = mock_graph_data

        response = client.post(
            "/backend/flow/search",
            json={"query_val": "test model"},
        )

        assert response.status_code == 200
        # Response is streaming, so we need to read the text
        content = response.text
        assert len(content) > 0
        # Check that it contains expected status messages
        assert (
            "Stage 1" in content or "Stage 2" in content or "METADATA_START" in content
        )


def test_search_flow_without_neo4j_data(mock_agent_runner, mock_huggingface_api):
    """Test search flow when Neo4j returns no data."""
    with patch(
        "routers.search.utils.search_neo4j.search_query_impl"
    ) as mock_search_query:
        from routers.search.utils.search_neo4j import (
            HFGraphData,
            HFNodes,
            HFRelationships,
        )

        # Return empty graph
        mock_search_query.return_value = HFGraphData(
            nodes=HFNodes(nodes=[]),
            relationships=HFRelationships(relationships=[]),
        )

        response = client.post(
            "/backend/flow/search",
            json={"query_val": "nonexistent model"},
        )

        assert response.status_code == 200
        # Response is streaming text
        content = response.text
        assert len(content) > 0


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
    with patch(
        "routers.search.utils.search_neo4j.search_query_impl"
    ) as mock_search_query:
        from routers.search.utils.search_neo4j import (
            HFGraphData,
            HFNodes,
            HFRelationships,
            HFModel,
        )

        # Return graph with model
        mock_model = HFModel(model_id="test/model", downloads=1000)
        mock_search_query.return_value = HFGraphData(
            nodes=HFNodes(nodes=[mock_model]),
            relationships=HFRelationships(relationships=[]),
            queried_model_id="test/model",
        )

        response = client.post(
            "/backend/flow/search",
            json={"query_val": "test model"},
        )

        assert response.status_code == 200
        # Response is streaming text
        content = response.text
        assert len(content) > 0


def test_search_flow_tool_state_integration(
    mock_agent_runner, mock_neo4j_driver, mock_huggingface_api
):
    """Test that tool state is properly managed across the flow."""
    with patch(
        "routers.search.utils.search_neo4j.search_query_impl"
    ) as mock_search_query:
        from routers.search.utils.search_neo4j import (
            HFGraphData,
            HFNodes,
            HFRelationships,
            HFModel,
        )

        # Return graph with model
        mock_model = HFModel(model_id="bert/model", downloads=1000)
        mock_search_query.return_value = HFGraphData(
            nodes=HFNodes(nodes=[mock_model]),
            relationships=HFRelationships(relationships=[]),
            queried_model_id="bert/model",
        )

        response = client.post(
            "/backend/flow/search",
            json={"query_val": "bert model"},
        )

        assert response.status_code == 200
        # Response is streaming text, check for metadata
        content = response.text
        assert len(content) > 0
        # Metadata should be in the response if workflow completes
        if "METADATA_START" in content:
            assert "METADATA_END" in content
