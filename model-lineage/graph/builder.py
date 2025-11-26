"""Build lineage graph from scraped data."""

import logging
from typing import List, Dict, Any, Optional

from graph.models import ModelNode, DatasetNode, Relationship, GraphData

logger = logging.getLogger(__name__)


class LineageGraphBuilder:
    """Builds a lineage graph from scraped model data."""

    def _convert_to_nodes(
        self,
        data_list: List[Dict[str, Any]],
        node_class: type,
        id_field: Optional[str],
        entity_type: str,
    ) -> List:
        """
        Generic method to convert dictionaries to node objects.

        Args:
            data_list: List of dictionaries to convert
            node_class: Pydantic model class to instantiate
            id_field: Field name for ID (for error messages)
            entity_type: Type name for error messages

        Returns:
            List of node objects
        """
        nodes = []
        for data in data_list:
            try:
                node = node_class(**data)
                nodes.append(node)
            except Exception as e:
                entity_id = data.get(id_field) if id_field else "unknown"
                logger.warning(
                    f"Failed to create {entity_type} node for {entity_id}: {e}"
                )
                continue
        return nodes

    def build_from_data(
        self,
        models: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        datasets: Optional[List[Dict[str, Any]]] = None,
    ) -> GraphData:
        """
        Build graph data structure from scraped models, datasets, and relationships.

        Args:
            models: List of model dictionaries
            relationships: List of relationship dictionaries
            datasets: Optional list of dataset dictionaries (if None, will infer from relationships)

        Returns:
            GraphData object
        """
        logger.info(
            f"Building graph from {len(models)} models, "
            f"{len(datasets) if datasets else 0} datasets, "
            f"and {len(relationships)} relationships"
        )

        # Convert models to ModelNode objects
        model_nodes = self._convert_to_nodes(models, ModelNode, "model_id", "model")

        # Convert datasets to DatasetNode objects
        dataset_nodes = []
        if datasets:
            dataset_nodes = self._convert_to_nodes(
                datasets, DatasetNode, "dataset_id", "dataset"
            )

        # If no datasets provided, infer from relationships
        if not datasets:
            dataset_ids = set()
            for rel in relationships:
                if rel.get("target_type") == "dataset":
                    dataset_ids.add(rel["target"])

            for dataset_id in dataset_ids:
                dataset_nodes.append(DatasetNode(dataset_id=dataset_id, tags=[]))

        # Convert relationships to Relationship objects
        relationship_objects = self._convert_to_nodes(
            relationships, Relationship, None, "relationship"
        )

        logger.info(
            f"Built graph with {len(model_nodes)} models, "
            f"{len(dataset_nodes)} datasets, "
            f"and {len(relationship_objects)} relationships"
        )

        return GraphData(
            models=model_nodes,
            datasets=dataset_nodes,
            relationships=relationship_objects,
        )
