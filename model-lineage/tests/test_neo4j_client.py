"""Tests for Neo4j client."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from graph.neo4j_client import Neo4jClient
from graph.models import ModelNode, DatasetNode, Relationship, GraphData


@pytest.fixture
def mock_driver():
    """Create a mock Neo4j driver."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = None
    driver.verify_connectivity.return_value = None
    return driver


@patch("graph.neo4j_client.GraphDatabase")
def test_neo4j_client_connection(mock_graph_db, mock_driver):
    """Test Neo4j client connection."""
    mock_graph_db.driver.return_value = mock_driver

    with patch("graph.neo4j_client.settings") as mock_settings:
        mock_settings.NEO4J_URI = "bolt://localhost:7687"
        mock_settings.NEO4J_USER = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"

        client = Neo4jClient()
        assert client.driver == mock_driver
        mock_driver.verify_connectivity.assert_called_once()


@patch("graph.neo4j_client.GraphDatabase")
def test_neo4j_client_close(mock_graph_db, mock_driver):
    """Test closing Neo4j connection."""
    mock_graph_db.driver.return_value = mock_driver

    with patch("graph.neo4j_client.settings") as mock_settings:
        mock_settings.NEO4J_URI = "bolt://localhost:7687"
        mock_settings.NEO4J_USER = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"

        client = Neo4jClient()
        client.close()
        mock_driver.close.assert_called_once()


@patch("graph.neo4j_client.GraphDatabase")
def test_clear_database(mock_graph_db, mock_driver):
    """Test clearing the database."""
    mock_graph_db.driver.return_value = mock_driver
    session = mock_driver.session.return_value.__enter__.return_value

    with patch("graph.neo4j_client.settings") as mock_settings:
        mock_settings.NEO4J_URI = "bolt://localhost:7687"
        mock_settings.NEO4J_USER = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"

        client = Neo4jClient()
        client.clear_database()
        session.run.assert_called_with("MATCH (n) DETACH DELETE n")


@patch("graph.neo4j_client.GraphDatabase")
def test_create_model_node(mock_graph_db, mock_driver):
    """Test creating a model node."""
    mock_graph_db.driver.return_value = mock_driver
    session = mock_driver.session.return_value.__enter__.return_value

    with patch("graph.neo4j_client.settings") as mock_settings:
        mock_settings.NEO4J_URI = "bolt://localhost:7687"
        mock_settings.NEO4J_USER = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"

        client = Neo4jClient()
        model = ModelNode(
            model_id="test/model",
            url="https://huggingface.co/test/model",
            author="test_author",
            downloads=1000,
        )

        client.create_model_node(model)

        # Verify session.run was called with correct query
        assert session.run.called
        call_args = session.run.call_args
        assert "MERGE (m:Model {model_id: $model_id})" in call_args[0][0]
        assert call_args[1]["model_id"] == "test/model"


@patch("graph.neo4j_client.GraphDatabase")
def test_create_dataset_node(mock_graph_db, mock_driver):
    """Test creating a dataset node."""
    mock_graph_db.driver.return_value = mock_driver
    session = mock_driver.session.return_value.__enter__.return_value

    with patch("graph.neo4j_client.settings") as mock_settings:
        mock_settings.NEO4J_URI = "bolt://localhost:7687"
        mock_settings.NEO4J_USER = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"

        client = Neo4jClient()
        dataset = DatasetNode(
            dataset_id="test/dataset", author="test_author", downloads=500
        )

        client.create_dataset_node(dataset)

        # Verify session.run was called
        assert session.run.called
        call_args = session.run.call_args
        assert "MERGE (d:Dataset {dataset_id: $dataset_id})" in call_args[0][0]
        assert call_args[1]["dataset_id"] == "test/dataset"


@patch("graph.neo4j_client.GraphDatabase")
def test_create_relationship_model_to_model(mock_graph_db, mock_driver):
    """Test creating a model-to-model relationship."""
    mock_graph_db.driver.return_value = mock_driver
    session = mock_driver.session.return_value.__enter__.return_value

    with patch("graph.neo4j_client.settings") as mock_settings:
        mock_settings.NEO4J_URI = "bolt://localhost:7687"
        mock_settings.NEO4J_USER = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"

        client = Neo4jClient()
        relationship = Relationship(
            source="model1",
            target="model2",
            relationship_type="BASED_ON",
            source_type="model",
            target_type="model",
        )

        client.create_relationship(relationship)

        # Verify session.run was called
        assert session.run.called
        call_args = session.run.call_args
        assert "MATCH" in call_args[0][0]
        assert "BASED_ON" in call_args[0][0]


@patch("graph.neo4j_client.GraphDatabase")
def test_create_relationship_model_to_dataset(mock_graph_db, mock_driver):
    """Test creating a model-to-dataset relationship."""
    mock_graph_db.driver.return_value = mock_driver
    session = mock_driver.session.return_value.__enter__.return_value

    with patch("graph.neo4j_client.settings") as mock_settings:
        mock_settings.NEO4J_URI = "bolt://localhost:7687"
        mock_settings.NEO4J_USER = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"

        client = Neo4jClient()
        relationship = Relationship(
            source="model1",
            target="dataset1",
            relationship_type="TRAINED_ON",
            source_type="model",
            target_type="dataset",
        )

        client.create_relationship(relationship)

        assert session.run.called
        call_args = session.run.call_args
        assert "TRAINED_ON" in call_args[0][0]


@patch("graph.neo4j_client.GraphDatabase")
def test_create_relationship_with_metadata(mock_graph_db, mock_driver):
    """Test creating a relationship with metadata."""
    mock_graph_db.driver.return_value = mock_driver
    session = mock_driver.session.return_value.__enter__.return_value

    with patch("graph.neo4j_client.settings") as mock_settings:
        mock_settings.NEO4J_URI = "bolt://localhost:7687"
        mock_settings.NEO4J_USER = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"

        client = Neo4jClient()
        relationship = Relationship(
            source="model1",
            target="model2",
            relationship_type="BASED_ON",
            source_type="model",
            target_type="model",
            metadata={"epochs": 10, "batch_size": 32},
        )

        client.create_relationship(relationship)

        assert session.run.called
        call_args = session.run.call_args
        # session.run(query, params) - params is second positional arg
        query = call_args[0][0]  # First positional arg is the query
        params = call_args[0][1]  # Second positional arg is the params dict

        assert "SET" in query  # Should have SET clause for metadata
        assert params["epochs"] == 10
        assert params["batch_size"] == 32
        assert params["source"] == "model1"
        assert params["target"] == "model2"


@patch("graph.neo4j_client.GraphDatabase")
def test_load_graph(mock_graph_db, mock_driver):
    """Test loading complete graph data."""
    mock_graph_db.driver.return_value = mock_driver
    session = mock_driver.session.return_value.__enter__.return_value

    with patch("graph.neo4j_client.settings") as mock_settings:
        mock_settings.NEO4J_URI = "bolt://localhost:7687"
        mock_settings.NEO4J_USER = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"

        client = Neo4jClient()

        models = [
            ModelNode(
                model_id="model1",
                url="https://huggingface.co/model1",
            )
        ]
        datasets = [DatasetNode(dataset_id="dataset1")]
        relationships = [
            Relationship(
                source="model1",
                target="dataset1",
                relationship_type="TRAINED_ON",
                source_type="model",
                target_type="dataset",
            )
        ]

        graph_data = GraphData(
            models=models, datasets=datasets, relationships=relationships
        )

        client.load_graph(graph_data)

        # Should create nodes and relationships
        assert session.run.call_count >= 3  # At least model, dataset, relationship


@patch("graph.neo4j_client.GraphDatabase")
def test_load_graph_with_failed_relationship(mock_graph_db, mock_driver):
    """Test loading graph when relationship creation fails."""
    mock_graph_db.driver.return_value = mock_driver
    session = mock_driver.session.return_value.__enter__.return_value

    with patch("graph.neo4j_client.settings") as mock_settings:
        mock_settings.NEO4J_URI = "bolt://localhost:7687"
        mock_settings.NEO4J_USER = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"

        client = Neo4jClient()

        # Make relationship creation raise an exception
        def failing_run(*args, **kwargs):
            if "TRAINED_ON" in str(args[0]):
                raise Exception("Relationship creation failed")
            return MagicMock()

        session.run.side_effect = failing_run

        models = [
            ModelNode(
                model_id="model1",
                url="https://huggingface.co/model1",
            )
        ]
        datasets = [DatasetNode(dataset_id="dataset1")]
        relationships = [
            Relationship(
                source="model1",
                target="dataset1",
                relationship_type="TRAINED_ON",
                source_type="model",
                target_type="dataset",
            )
        ]

        graph_data = GraphData(
            models=models, datasets=datasets, relationships=relationships
        )

        # Should not raise, just log warning
        client.load_graph(graph_data)


@patch("graph.neo4j_client.GraphDatabase")
def test_get_model_lineage(mock_graph_db, mock_driver):
    """Test getting model lineage."""
    mock_graph_db.driver.return_value = mock_driver
    session = mock_driver.session.return_value.__enter__.return_value

    # Mock result
    mock_record = MagicMock()
    mock_record.__getitem__.return_value = "mock_path"
    mock_result = [mock_record]
    session.run.return_value = mock_result

    with patch("graph.neo4j_client.settings") as mock_settings:
        mock_settings.NEO4J_URI = "bolt://localhost:7687"
        mock_settings.NEO4J_USER = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"

        client = Neo4jClient()
        result = client.get_model_lineage("test/model", depth=3)

        assert result["model_id"] == "test/model"
        assert result["depth"] == 3
        assert "paths" in result
        assert session.run.called


@patch("graph.neo4j_client.GraphDatabase")
def test_get_statistics(mock_graph_db, mock_driver):
    """Test getting graph statistics."""
    mock_graph_db.driver.return_value = mock_driver
    session = mock_driver.session.return_value.__enter__.return_value

    def mock_run(query, **kwargs):
        mock_result = MagicMock()
        if "relationship_types" in query:
            # Return iterable of dict-like records
            # Use a class that can be converted to dict
            class MockRecord:
                def __init__(self, data):
                    self._data = data

                def __getitem__(self, key):
                    return self._data[key]

                def keys(self):
                    return self._data.keys()

                def items(self):
                    return self._data.items()

                def __iter__(self):
                    return iter(self._data)

            record1 = MockRecord({"rel_type": "BASED_ON", "count": 10})
            record2 = MockRecord({"rel_type": "TRAINED_ON", "count": 5})
            mock_result.__iter__.return_value = iter([record1, record2])
        else:
            # Return result with single() method
            mock_single_result = MagicMock()
            mock_single_result.single.return_value = {"count": 100}
            mock_result = mock_single_result
        return mock_result

    session.run.side_effect = mock_run

    with patch("graph.neo4j_client.settings") as mock_settings:
        mock_settings.NEO4J_URI = "bolt://localhost:7687"
        mock_settings.NEO4J_USER = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"

        client = Neo4jClient()
        stats = client.get_statistics()

        assert "model_count" in stats
        assert "dataset_count" in stats
        assert "relationship_count" in stats
        assert "relationship_types" in stats
        assert isinstance(stats["relationship_types"], list)
