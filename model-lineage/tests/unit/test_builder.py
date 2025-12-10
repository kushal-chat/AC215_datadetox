"""Tests for graph builder."""

from __future__ import annotations


from graph.builder import LineageGraphBuilder
from graph.models import GraphData


def test_build_from_data_with_all_inputs():
    """Test building graph with models, datasets, and relationships."""
    builder = LineageGraphBuilder()

    models = [
        {
            "model_id": "model1",
            "url": "https://huggingface.co/model1",
            "author": "author1",
            "downloads": 1000,
        },
        {
            "model_id": "model2",
            "url": "https://huggingface.co/model2",
            "author": "author2",
            "downloads": 2000,
        },
    ]

    datasets = [{"dataset_id": "dataset1", "author": "author1", "downloads": 500}]

    relationships = [
        {
            "source": "model1",
            "target": "model2",
            "relationship_type": "based_on",
            "source_type": "model",
            "target_type": "model",
        },
        {
            "source": "model1",
            "target": "dataset1",
            "relationship_type": "trained_on",
            "source_type": "model",
            "target_type": "dataset",
        },
    ]

    graph = builder.build_from_data(models, relationships, datasets)

    assert isinstance(graph, GraphData)
    assert len(graph.models) == 2
    assert len(graph.datasets) == 1
    assert len(graph.relationships) == 2
    assert graph.models[0].model_id == "model1"
    assert graph.datasets[0].dataset_id == "dataset1"


def test_build_from_data_without_datasets():
    """Test building graph without explicit datasets (inferred from relationships)."""
    builder = LineageGraphBuilder()

    models = [
        {
            "model_id": "model1",
            "url": "https://huggingface.co/model1",
        }
    ]

    relationships = [
        {
            "source": "model1",
            "target": "dataset1",
            "relationship_type": "trained_on",
            "source_type": "model",
            "target_type": "dataset",
        }
    ]

    graph = builder.build_from_data(models, relationships, datasets=None)

    assert len(graph.models) == 1
    assert len(graph.datasets) == 1  # Inferred from relationship
    assert graph.datasets[0].dataset_id == "dataset1"
    assert graph.datasets[0].tags == []  # Default tags


def test_build_from_data_with_invalid_model():
    """Test that invalid model data is skipped with warning."""
    builder = LineageGraphBuilder()

    models = [
        {
            "model_id": "valid_model",
            "url": "https://huggingface.co/valid_model",
        },
        {
            # Missing required field 'url'
            "model_id": "invalid_model",
        },
    ]

    relationships = []

    graph = builder.build_from_data(models, relationships)

    # Only valid model should be included
    assert len(graph.models) == 1
    assert graph.models[0].model_id == "valid_model"


def test_build_from_data_with_invalid_dataset():
    """Test that invalid dataset data is skipped."""
    builder = LineageGraphBuilder()

    models = []

    datasets = [
        {"dataset_id": "valid_dataset"},
        {
            # Missing required field 'dataset_id'
            "author": "author1",
        },
    ]

    relationships = []

    graph = builder.build_from_data(models, relationships, datasets)

    # Only valid dataset should be included
    assert len(graph.datasets) == 1
    assert graph.datasets[0].dataset_id == "valid_dataset"


def test_build_from_data_with_invalid_relationship():
    """Test that invalid relationship data is skipped."""
    builder = LineageGraphBuilder()

    models = [
        {
            "model_id": "model1",
            "url": "https://huggingface.co/model1",
        }
    ]

    relationships = [
        {
            "source": "model1",
            "target": "model2",
            "relationship_type": "based_on",
            "source_type": "model",
            "target_type": "model",
        },
        {
            # Missing required field 'target'
            "source": "model1",
            "relationship_type": "based_on",
            "source_type": "model",
            "target_type": "model",
        },
    ]

    graph = builder.build_from_data(models, relationships)

    # Only valid relationship should be included
    assert len(graph.relationships) == 1
    assert graph.relationships[0].source == "model1"


def test_build_from_data_empty_inputs():
    """Test building graph with empty inputs."""
    builder = LineageGraphBuilder()

    graph = builder.build_from_data([], [], [])

    assert len(graph.models) == 0
    assert len(graph.datasets) == 0
    assert len(graph.relationships) == 0


def test_build_from_data_model_node_attributes():
    """Test that ModelNode attributes are correctly set."""
    builder = LineageGraphBuilder()

    models = [
        {
            "model_id": "test/model",
            "url": "https://huggingface.co/test/model",
            "author": "test_author",
            "downloads": 1000,
            "likes": 50,
            "tags": ["nlp", "bert"],
            "library_name": "transformers",
            "pipeline_tag": "text-classification",
            "private": False,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
        }
    ]

    graph = builder.build_from_data(models, [])

    model = graph.models[0]
    assert model.model_id == "test/model"
    assert model.author == "test_author"
    assert model.downloads == 1000
    assert model.likes == 50
    assert model.tags == ["nlp", "bert"]
    assert model.library_name == "transformers"
    assert model.pipeline_tag == "text-classification"
    assert model.private is False
