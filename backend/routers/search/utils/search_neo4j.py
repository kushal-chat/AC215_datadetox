"""Tools to interface with the Neo4j database for HuggingFace model lineage."""

from __future__ import annotations

import os
import logging
from typing import Optional, Union

import neo4j
from agents import function_tool
from pydantic import BaseModel, ConfigDict

from .tool_state import set_tool_result

logger = logging.getLogger(__name__)

# Neo4j connection configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
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
    queried_model_id: Optional[str] = None  # The model ID that was queried


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
    """Search for all models in the Neo4j database (most downloaded)."""
    res, summary, _ = driver.execute_query(
        "MATCH (n:Model) RETURN n ORDER BY n.downloads DESC",
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
    """Search for all datasets in the Neo4j database (most downloaded)."""
    res, summary, _ = driver.execute_query(
        "MATCH (n:Dataset) RETURN n ORDER BY n.downloads DESC",
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
    """Get lineage tree for a given model: queried model + prioritized upstreams + capped downstreams (max 10 related)."""
    logger.info(f"Searching Neo4j for model: {model_id}")
    MAX_RELATED = 10  # cap related models to avoid overly large trees

    # First, get the queried model itself
    root_query = """
        MATCH (root:Model {model_id: $model_id})
        RETURN root
    """
    root_res, _, _ = driver.execute_query(
        root_query,
        model_id=model_id,
        routing_=neo4j.RoutingControl.READ,
    )

    if not root_res:
        logger.warning(f"Model {model_id} not found in Neo4j")
        return HFGraphData(
            nodes=HFNodes(nodes=[]),
            relationships=HFRelationships(relationships=[]),
        )

    root_node_dict = root_res[0].data()["root"]
    root_model = _make_entity(root_node_dict)
    if not isinstance(root_model, HFModel):
        logger.error(f"Root node {model_id} is not a Model")
        return HFGraphData(
            nodes=HFNodes(nodes=[]),
            relationships=HFRelationships(relationships=[]),
        )

    # Get upstream models first (consume most of the limit)
    # Direction: upstream -> queried_model (INCOMING relationships to queried model)
    upstream_query = """
        MATCH (root:Model {model_id: $model_id})<-[r:BASED_ON|FINE_TUNED|FINETUNED|ADAPTERS|MERGES|QUANTIZATIONS|TRAINED_ON]-(upstream:Model)
        RETURN upstream, type(r) as rel_type
        ORDER BY upstream.downloads DESC
        LIMIT $limit
    """
    upstream_res, _, _ = driver.execute_query(
        upstream_query,
        model_id=model_id,
        limit=MAX_RELATED,
        routing_=neo4j.RoutingControl.READ,
    )

    # Remaining budget for downstream (queried_model -> downstream)
    remaining_budget = max(0, MAX_RELATED - len(upstream_res))
    if remaining_budget > 0:
        downstream_query = """
            MATCH (root:Model {model_id: $model_id})-[r:BASED_ON|FINE_TUNED|FINETUNED|ADAPTERS|MERGES|QUANTIZATIONS|TRAINED_ON]->(downstream:Model)
            RETURN downstream, type(r) as rel_type
            ORDER BY downstream.downloads DESC
            LIMIT $limit
        """
        downstream_res, _, _ = driver.execute_query(
            downstream_query,
            model_id=model_id,
            limit=remaining_budget,
            routing_=neo4j.RoutingControl.READ,
        )
    else:
        downstream_res = []

    def _ensure_entity(node: HFModel | HFDataset | dict) -> HFModel | HFDataset:
        if isinstance(node, (HFModel, HFDataset)):
            return node
        return _make_entity(node)

    # Collect all nodes (use dict to avoid duplicates based on model_id)
    all_nodes_dict = {root_model.model_id: root_model}

    # Build relationships: only direct connections to/from queried model
    relationships = []

    # Process upstream models (all of them) and build relationships
    # Upstream relationships: upstream -> queried_model
    for record in upstream_res:
        upstream_dict = record.data()["upstream"]
        upstream_model = _ensure_entity(upstream_dict)
        if isinstance(upstream_model, HFModel):
            all_nodes_dict[upstream_model.model_id] = upstream_model
            rel_type = record.data()["rel_type"]
            relationships.append(
                HFRelationship(
                    source=upstream_model,
                    relationship=rel_type,
                    target=root_model,
                )
            )

    # Process downstream models (max 5) and build relationships
    # Downstream relationships: queried_model -> downstream
    for record in downstream_res:
        downstream_dict = record.data()["downstream"]
        downstream_model = _ensure_entity(downstream_dict)
        if isinstance(downstream_model, HFModel):
            all_nodes_dict[downstream_model.model_id] = downstream_model
            rel_type = record.data()["rel_type"]
            relationships.append(
                HFRelationship(
                    source=root_model,
                    relationship=rel_type,
                    target=downstream_model,
                )
            )

    all_nodes = list(all_nodes_dict.values())

    upstream_count = sum(1 for r in relationships if r.target.model_id == model_id)
    downstream_count = sum(1 for r in relationships if r.source.model_id == model_id)

    logger.info(
        f"Found {upstream_count} upstream models, "
        f"{downstream_count} downstream models, "
        f"{len(all_nodes)} total nodes, "
        f"{len(relationships)} relationships"
    )

    result = HFGraphData(
        nodes=HFNodes(nodes=all_nodes),
        relationships=HFRelationships(relationships=relationships),
        queried_model_id=model_id,
    )

    # Store the result in request-scoped state for later retrieval
    set_tool_result("search_neo4j", result)

    return result
