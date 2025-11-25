"""Tools to interface with the Neo4j database for HuggingFace model lineage."""

from __future__ import annotations

import logging
from typing import Optional, Union

import neo4j
from agents import function_tool
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# Neo4j connection configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")
RELATIONSHIP_FILTER = (
    "BASED_ON|FINE_TUNED|FINETUNED|ADAPTERS|MERGES|QUANTIZATIONS|TRAINED_ON"
)

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


class HFModel(BaseModel):
    """HuggingFace model representation."""

    model_config = ConfigDict(extra="ignore")
    model_id: str
    downloads: Optional[int] = None
    pipeline_tag: Optional[str] = None
    created_at: Optional[str] = None
    library_name: Optional[str] = None
    url: Optional[str] = None
    likes: Optional[int] = None
    tags: list[str] = []


class HFDataset(BaseModel):
    """HuggingFace dataset representation."""

    dataset_id: str
    tags: list[str] = []


class HFRelationship(BaseModel):
    """Relationship between two HuggingFace entities."""

    source: Union[HFModel, HFDataset]
    relationship: str
    target: Union[HFModel, HFDataset]


class HFNodes(BaseModel):
    """Collection of HuggingFace nodes."""

    nodes: list[Union[HFModel, HFDataset]]


class HFRelationships(BaseModel):
    """Collection of HuggingFace relationships."""

    relationships: list[HFRelationship]


class HFGraphData(BaseModel):
    """Complete graph data structure."""

    nodes: HFNodes
    relationships: HFRelationships


def _parse_node(
    node_data: dict, entity_class: type[HFModel | HFDataset]
) -> HFModel | HFDataset | None:
    """Parse node data into the appropriate entity class."""
    try:
        return entity_class(**node_data)
    except Exception as e:
        logger.warning(f"Failed to parse node: {e}")
        return None


def _log_query_summary(summary: neo4j.QueryResultSummary, record_count: int) -> None:
    """Log query execution summary."""
    logger.info(
        f"Query '{summary.query}' returned {record_count} records in "
        f"{summary.result_available_after} ms"
    )


@function_tool
def search_models() -> HFNodes:
    """Search for all models in the Neo4j database."""
    res, summary, _ = driver.execute_query(
        "MATCH (n:Model) RETURN n",
        routing_=neo4j.RoutingControl.READ,
    )

    nodes = [
        model
        for node in res
        if (model := _parse_node(node.data()["n"], HFModel)) is not None
    ]

    _log_query_summary(summary, len(res))
    return HFNodes(nodes=nodes)


@function_tool
def search_datasets() -> HFNodes:
    """Search for all datasets in the Neo4j database."""
    res, summary, _ = driver.execute_query(
        "MATCH (n:Dataset) RETURN n",
        routing_=neo4j.RoutingControl.READ,
    )

    nodes = [
        dataset
        for node in res
        if (dataset := _parse_node(node.data()["n"], HFDataset)) is not None
    ]

    _log_query_summary(summary, len(res))
    return HFNodes(nodes=nodes)


def _make_entity(node_dict: dict) -> HFModel | HFDataset:
    """Create an entity instance from a node dictionary."""
    if "model_id" in node_dict:
        return HFModel(**node_dict)
    if "dataset_id" in node_dict:
        return HFDataset(**node_dict)
    raise ValueError(f"Cannot determine entity type from: {node_dict}")


@function_tool
def search_query(model_id: str) -> HFGraphData:
    """Get the complete lineage tree for a given model."""
    query = """
        MATCH (root:Model {model_id: $model_id})
        CALL apoc.path.subgraphAll(root, {
          relationshipFilter: $relationship_filter
        })
        YIELD nodes, relationships
        RETURN nodes, relationships
    """
    res, summary, _ = driver.execute_query(
        query,
        model_id=model_id,
        relationship_filter=RELATIONSHIP_FILTER,
        routing_=neo4j.RoutingControl.READ,
    )

    logger.info("Searching Neo4j.")

    if not res:
        return HFGraphData(
            nodes=HFNodes(nodes=[]),
            relationships=HFRelationships(relationships=[]),
        )

    data = res[0].data()

    def _ensure_entity(node: HFModel | HFDataset | dict) -> HFModel | HFDataset:
        if isinstance(node, (HFModel, HFDataset)):
            return node
        return _make_entity(node)

    node_entities = [_ensure_entity(node_dict) for node_dict in data.get("nodes", [])]
    MAX_COUNT = 5
    limited_nodes = node_entities[:MAX_COUNT]

    relationships = [
        HFRelationship(
            source=_ensure_entity(src_dict),
            relationship=rel_type,
            target=_ensure_entity(tgt_dict),
        )
        for src_dict, rel_type, tgt_dict in data["relationships"]
    ]

    _log_query_summary(summary, len(res))
    return HFGraphData(
        nodes=HFNodes(nodes=limited_nodes),
        relationships=HFRelationships(relationships=relationships[:MAX_COUNT]),
    )
