"""Hypothesis-based tests for search_neo4j module."""

from __future__ import annotations


from hypothesis import given, strategies as st
from unittest.mock import MagicMock, patch
from routers.search.utils.search_neo4j import (
    _log_query_summary,
)
from routers.search.utils.search_neo4j import (
    HFModel,
    HFDataset,
    HFRelationship,
    HFNodes,
    HFRelationships,
    HFGraphData,
    _make_entity,
    _parse_node,
)


def model_strategy() -> st.SearchStrategy[dict]:
    """
    Generate valid HFModel data with optional fields
    Returns:
        SearchStrategy
    """
    return st.fixed_dictionaries(
        {
            "model_id": st.text(min_size=1, max_size=100),
            "downloads": st.one_of(st.none(), st.integers(min_value=0)),
            "pipeline_tag": st.one_of(st.none(), st.text(min_size=1, max_size=50)),
            "created_at": st.one_of(
                st.none(), st.datetimes().map(lambda d: d.isoformat())
            ),
            "library_name": st.one_of(st.none(), st.text(min_size=1, max_size=50)),
            "url": st.one_of(st.none(), st.text(min_size=1, max_size=100)),
            "likes": st.one_of(st.none(), st.integers(min_value=0)),
            "tags": st.lists(st.text(min_size=1, max_size=30), max_size=20),
        },
    )


def dataset_strategy() -> st.SearchStrategy[dict]:
    """
    Generate valid HFDataset data with optional fields
    Returns:
        SearchStrategy
    """
    return st.fixed_dictionaries(
        {
            "dataset_id": st.text(min_size=1, max_size=100),
            "tags": st.lists(st.text(min_size=1, max_size=30), max_size=20),
        }
    )


@given(model_strategy())
def test_hfmodel_creation(model_data: dict) -> None:
    model = HFModel(**model_data)
    assert model.model_id == model_data["model_id"]
    assert model.downloads == model_data.get("downloads")
    assert model.tags == model_data.get("tags", [])


@given(dataset_strategy())
def test_hfdataset_creation(dataset_data: dict) -> None:
    dataset = HFDataset(**dataset_data)
    assert dataset.dataset_id == dataset_data["dataset_id"]
    assert dataset.tags == dataset_data["tags"]


@given(model_strategy(), st.text(min_size=1, max_size=50), model_strategy())
def test_hfrelationship_creation(
    source_data: dict, relationship: str, target_data: dict
) -> None:
    source = HFModel(**source_data)
    target = HFModel(**target_data)
    rel = HFRelationship(
        source=source,
        relationship=relationship,
        target=target,
    )
    assert rel.source == source
    assert rel.relationship == relationship
    assert rel.target == target


@given(
    st.one_of(model_strategy(), dataset_strategy()),
    st.text(min_size=1, max_size=50),
    st.one_of(model_strategy(), dataset_strategy()),
)
def test_hfrelationship_mixed_types(
    source_data: dict, relationship: str, target_data: dict
) -> None:
    source = (
        HFModel(**source_data)
        if "model_id" in source_data
        else HFDataset(**source_data)
    )
    target = (
        HFModel(**target_data)
        if "model_id" in target_data
        else HFDataset(**target_data)
    )
    rel = HFRelationship(
        source=source,
        relationship=relationship,
        target=target,
    )
    assert rel.source == source
    assert rel.target == target


@given(st.lists(model_strategy(), max_size=10))
def test_hfnodes_creation(models_data: list[dict]) -> None:
    models = [HFModel(**data) for data in models_data]
    nodes = HFNodes(nodes=models)
    assert len(nodes.nodes) == len(models)
    assert all(isinstance(node, HFModel) for node in nodes.nodes)


@given(
    st.lists(
        st.tuples(
            model_strategy(),
            st.text(min_size=1, max_size=50),
            model_strategy(),
        ),
        max_size=10,
    )
)
def test_hfrelationships_creation(
    relationships_data: list[tuple[dict, str, dict]],
) -> None:
    relationships = [
        HFRelationship(
            source=HFModel(**src),
            relationship=rel,
            target=HFModel(**tgt),
        )
        for src, rel, tgt in relationships_data
    ]
    hf_rels = HFRelationships(relationships=relationships)
    assert len(hf_rels.relationships) == len(relationships)


@given(model_strategy(), dataset_strategy())
def test_hfgraphdata_creation(model_data: dict, dataset_data: dict) -> None:
    models = [HFModel(**model_data)]
    datasets = [HFDataset(**dataset_data)]
    nodes = HFNodes(nodes=models + datasets)

    relationships = [
        HFRelationship(
            source=models[0],
            relationship="TRAINED_ON",
            target=datasets[0],
        )
    ]
    rels = HFRelationships(relationships=relationships)

    graph = HFGraphData(nodes=nodes, relationships=rels)
    assert len(graph.nodes.nodes) == 2
    assert len(graph.relationships.relationships) == 1


@given(model_strategy())
def test_make_entity_model(model_data: dict) -> None:
    """Test _make_entity correctly creates HFModel from dict"""
    entity = _make_entity(model_data)
    assert isinstance(entity, HFModel)
    assert entity.model_id == model_data["model_id"]


@given(dataset_strategy())
def test_make_entity_dataset(dataset_data: dict) -> None:
    """Test _make_entity correctly creates HFDataset from dict"""
    entity = _make_entity(dataset_data)
    assert isinstance(entity, HFDataset)
    assert entity.dataset_id == dataset_data["dataset_id"]


@given(st.dictionaries(st.text(), st.text()))
def test_make_entity_invalid(invalid_data: dict) -> None:
    """Test _make_entity raises ValueError for invalid data"""
    if "model_id" not in invalid_data and "dataset_id" not in invalid_data:
        try:
            _make_entity(invalid_data)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


@given(model_strategy())
def test_parse_node_model(model_data: dict) -> None:
    """Test _parse_node correctly parses model data"""
    from routers.search.utils.search_neo4j import HFModel

    result = _parse_node(model_data, HFModel)
    assert result is not None
    assert isinstance(result, HFModel)
    assert result.model_id == model_data["model_id"]


@given(dataset_strategy())
def test_parse_node_dataset(dataset_data: dict) -> None:
    """Test _parse_node correctly parses dataset data"""
    from routers.search.utils.search_neo4j import HFDataset

    result = _parse_node(dataset_data, HFDataset)
    assert result is not None
    assert isinstance(result, HFDataset)
    assert result.dataset_id == dataset_data["dataset_id"]


@given(st.dictionaries(st.text(), st.integers()))
def test_parse_node_invalid(invalid_data: dict) -> None:
    """Test _parse_node returns None for invalid data"""
    from routers.search.utils.search_neo4j import HFModel

    result = _parse_node(invalid_data, HFModel)
    assert result is None


@patch("routers.search.utils.search_neo4j.logger")
def test_log_query_summary(mock_logger):
    """Test _log_query_summary function."""
    mock_summary = MagicMock()
    mock_summary.query = "MATCH (n) RETURN n"
    mock_summary.result_available_after = 15

    _log_query_summary(mock_summary, 5)

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args[0][0]
    assert "5 records" in call_args
    assert "15 ms" in call_args


def test_make_entity_raises_value_error():
    """Test _make_entity raises ValueError for invalid data."""
    invalid_data = {"invalid": "data"}

    try:
        _make_entity(invalid_data)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot determine entity type" in str(e)
