"""Tests for graph models."""

from __future__ import annotations

from hypothesis import given, strategies as st

from graph.models import ModelNode, DatasetNode, Relationship, GraphData


def test_model_node_creation():
    """Test creating a ModelNode with required fields."""
    model = ModelNode(
        model_id="test/model",
        url="https://huggingface.co/test/model",
    )
    assert model.model_id == "test/model"
    assert model.url == "https://huggingface.co/test/model"
    assert model.private is False
    assert model.tags == []


def test_model_node_with_optional_fields():
    """Test creating a ModelNode with all optional fields."""
    model = ModelNode(
        model_id="test/model",
        url="https://huggingface.co/test/model",
        author="test_author",
        downloads=1000,
        likes=50,
        tags=["nlp", "bert"],
        library_name="transformers",
        pipeline_tag="text-classification",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-02T00:00:00Z",
    )
    assert model.author == "test_author"
    assert model.downloads == 1000
    assert model.likes == 50
    assert model.tags == ["nlp", "bert"]
    assert model.library_name == "transformers"
    assert model.pipeline_tag == "text-classification"


def test_dataset_node_creation():
    """Test creating a DatasetNode."""
    dataset = DatasetNode(dataset_id="test/dataset")
    assert dataset.dataset_id == "test/dataset"
    assert dataset.tags == []


def test_dataset_node_with_optional_fields():
    """Test creating a DatasetNode with optional fields."""
    dataset = DatasetNode(
        dataset_id="test/dataset",
        author="test_author",
        downloads=500,
        tags=["nlp", "text"],
    )
    assert dataset.author == "test_author"
    assert dataset.downloads == 500
    assert dataset.tags == ["nlp", "text"]


def test_relationship_creation():
    """Test creating a Relationship."""
    rel = Relationship(
        source="model1",
        target="model2",
        relationship_type="based_on",
        source_type="model",
        target_type="model",
    )
    assert rel.source == "model1"
    assert rel.target == "model2"
    assert rel.relationship_type == "based_on"
    assert rel.source_type == "model"
    assert rel.target_type == "model"
    assert rel.metadata is None


def test_relationship_with_metadata():
    """Test creating a Relationship with metadata."""
    rel = Relationship(
        source="model1",
        target="dataset1",
        relationship_type="trained_on",
        source_type="model",
        target_type="dataset",
        metadata={"epochs": 10, "batch_size": 32},
    )
    assert rel.metadata == {"epochs": 10, "batch_size": 32}


def test_graph_data_creation():
    """Test creating GraphData."""
    models = [
        ModelNode(model_id="model1", url="https://hf.co/model1"),
        ModelNode(model_id="model2", url="https://hf.co/model2"),
    ]
    datasets = [DatasetNode(dataset_id="dataset1")]
    relationships = [
        Relationship(
            source="model1",
            target="model2",
            relationship_type="based_on",
            source_type="model",
            target_type="model",
        )
    ]

    graph = GraphData(models=models, datasets=datasets, relationships=relationships)
    assert len(graph.models) == 2
    assert len(graph.datasets) == 1
    assert len(graph.relationships) == 1
    assert graph.metadata is None


def test_graph_data_with_metadata():
    """Test creating GraphData with metadata."""
    graph = GraphData(
        models=[],
        datasets=[],
        relationships=[],
        metadata={"version": "1.0", "timestamp": "2024-01-01"},
    )
    assert graph.metadata == {"version": "1.0", "timestamp": "2024-01-01"}


# Hypothesis-based property tests
@given(
    model_id=st.text(min_size=1, max_size=100),
    url=st.text(min_size=1, max_size=200),
    downloads=st.one_of(st.none(), st.integers(min_value=0)),
    likes=st.one_of(st.none(), st.integers(min_value=0)),
)
def test_model_node_properties(model_id: str, url: str, downloads, likes):
    """Property-based test for ModelNode creation."""
    model = ModelNode(
        model_id=model_id,
        url=url,
        downloads=downloads,
        likes=likes,
    )
    assert model.model_id == model_id
    assert model.url == url
    assert model.downloads == downloads
    assert model.likes == likes


@given(
    dataset_id=st.text(min_size=1, max_size=100),
    downloads=st.one_of(st.none(), st.integers(min_value=0)),
)
def test_dataset_node_properties(dataset_id: str, downloads):
    """Property-based test for DatasetNode creation."""
    dataset = DatasetNode(dataset_id=dataset_id, downloads=downloads)
    assert dataset.dataset_id == dataset_id
    assert dataset.downloads == downloads
