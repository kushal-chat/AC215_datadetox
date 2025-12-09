"""Tests for the client router endpoint."""

from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from main import app


def test_search_endpoint_success():
    """Test the search endpoint with successful agent execution."""
    client = TestClient(app)

    # Mock the Runner to avoid actual agent calls
    mock_result = MagicMock()
    mock_result.final_output_as.return_value = "Test response"

    with patch("routers.client.Runner") as mock_runner:
        mock_runner.run = AsyncMock(return_value=mock_result)

        # Mock tool_state functions
        with (
            patch("routers.client.set_request_context") as mock_set_context,
            patch(
                "routers.client.get_tool_result", return_value=None
            ) as mock_get_result,
        ):
            response = client.post(
                "/backend/flow/search", json={"query_val": "test query"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "result" in data
            assert data["result"] == "Test response"
            assert "neo4j_data" not in data  # No neo4j result in this case
            mock_set_context.assert_called_once()
            # Check that get_tool_result was called with correct args (Request object may differ)
            assert mock_get_result.called
            call_args = mock_get_result.call_args[0]
            assert call_args[0] == "search_neo4j"
            assert hasattr(call_args[1], "state")  # Verify it's a Request-like object


def test_search_endpoint_with_neo4j_data():
    """Test the search endpoint when neo4j_data is available."""
    client = TestClient(app)

    # Mock the Runner
    mock_result = MagicMock()
    mock_result.final_output_as.return_value = "Test response"

    # Mock neo4j result
    mock_neo4j_data = MagicMock()
    mock_neo4j_data.model_dump.return_value = {"nodes": [], "relationships": []}

    with patch("routers.client.Runner") as mock_runner:
        mock_runner.run = AsyncMock(return_value=mock_result)

        with (
            patch("routers.client.set_request_context"),
            patch("routers.client.get_tool_result", return_value=mock_neo4j_data),
        ):
            response = client.post(
                "/backend/flow/search", json={"query_val": "test query"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "result" in data
            assert "neo4j_data" in data
            assert data["neo4j_data"] == {"nodes": [], "relationships": []}


def test_search_endpoint_query_validation():
    """Test that the endpoint validates query input."""
    client = TestClient(app)

    # Missing query_val
    response = client.post("/backend/flow/search", json={})
    assert response.status_code == 422  # Validation error

    # Invalid JSON - use content parameter instead of data to avoid deprecation warning
    response = client.post(
        "/backend/flow/search",
        content="invalid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
