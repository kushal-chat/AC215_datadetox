"""Neo4j client for graph database operations."""

import logging
from typing import Dict, Any, Optional
from neo4j import GraphDatabase, Driver

from config.settings import settings
from graph.models import ModelNode, DatasetNode, Relationship, GraphData

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Client for Neo4j graph database operations."""

    def __init__(self):
        self.driver: Optional[Driver] = None
        self._connect()

    def _connect(self):
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            # Verify connection
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {settings.NEO4J_URI}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared Neo4j database")

    def create_model_node(self, model: ModelNode):
        """Create a model node in Neo4j."""
        query = """
        MERGE (m:Model {model_id: $model_id})
        SET m.author = $author,
            m.downloads = $downloads,
            m.likes = $likes,
            m.tags = $tags,
            m.library_name = $library_name,
            m.pipeline_tag = $pipeline_tag,
            m.private = $private,
            m.url = $url,
            m.created_at = $created_at,
            m.updated_at = $updated_at
        """

        with self.driver.session() as session:
            session.run(query, **model.model_dump())

    def create_dataset_node(self, dataset: DatasetNode):
        """Create a dataset node in Neo4j."""
        query = """
        MERGE (d:Dataset {dataset_id: $dataset_id})
        SET d.author = $author,
            d.downloads = $downloads,
            d.tags = $tags
        """

        with self.driver.session() as session:
            session.run(query, **dataset.model_dump())

    def create_relationship(self, relationship: Relationship):
        """Create a relationship between nodes."""
        # Determine source and target node types
        source_label = "Model" if relationship.source_type == "model" else "Dataset"
        target_label = "Model" if relationship.target_type == "model" else "Dataset"

        source_id_field = (
            "model_id" if relationship.source_type == "model" else "dataset_id"
        )
        target_id_field = (
            "model_id" if relationship.target_type == "model" else "dataset_id"
        )

        # Map relationship type to Neo4j relationship type
        rel_type = relationship.relationship_type.upper()

        query = f"""
        MATCH (source:{source_label} {{{source_id_field}: $source}})
        MATCH (target:{target_label} {{{target_id_field}: $target}})
        MERGE (source)-[r:{rel_type}]->(target)
        """

        if relationship.metadata:
            # Add metadata as relationship properties
            set_clauses = [f"r.{k} = ${k}" for k in relationship.metadata.keys()]
            if set_clauses:
                query += f" SET {', '.join(set_clauses)}"

        params = {
            "source": relationship.source,
            "target": relationship.target,
            **(relationship.metadata or {}),
        }

        with self.driver.session() as session:
            session.run(query, params)

    def load_graph(self, graph_data: GraphData):
        """Load complete graph data into Neo4j."""
        logger.info(
            f"Loading {len(graph_data.models)} models, "
            f"{len(graph_data.datasets)} datasets, "
            f"and {len(graph_data.relationships)} relationships into Neo4j"
        )

        # Create model nodes
        for model in graph_data.models:
            self.create_model_node(model)

        # Create dataset nodes
        for dataset in graph_data.datasets:
            self.create_dataset_node(dataset)

        # Create relationships
        for relationship in graph_data.relationships:
            try:
                self.create_relationship(relationship)
            except Exception as e:
                logger.warning(
                    f"Failed to create relationship {relationship.source} -> {relationship.target}: {e}"
                )

        logger.info("Graph loaded successfully")

    def get_model_lineage(self, model_id: str, depth: int = 3) -> Dict[str, Any]:
        """
        Get the lineage tree for a specific model.

        Args:
            model_id: Model ID to get lineage for
            depth: Maximum depth to traverse

        Returns:
            Dictionary with lineage information
        """
        query = f"""
        MATCH path = (m:Model {{model_id: $model_id}})-[*1..{depth}]-(related)
        RETURN path
        LIMIT 100
        """

        with self.driver.session() as session:
            result = session.run(query, model_id=model_id)
            paths = [record["path"] for record in result]

            return {"model_id": model_id, "paths": paths, "depth": depth}

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the graph."""
        queries = {
            "model_count": "MATCH (m:Model) RETURN count(m) as count",
            "dataset_count": "MATCH (d:Dataset) RETURN count(d) as count",
            "relationship_count": "MATCH ()-[r]->() RETURN count(r) as count",
            "relationship_types": """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
            """,
        }

        stats = {}
        with self.driver.session() as session:
            for key, query in queries.items():
                result = session.run(query)
                if key == "relationship_types":
                    stats[key] = [dict(record) for record in result]
                else:
                    stats[key] = result.single()["count"]

        return stats
