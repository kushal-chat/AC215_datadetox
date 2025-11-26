"""Build lineage graph from scraped data."""

import logging
from typing import List, Dict, Any, Optional

from graph.models import ModelNode, DatasetNode, Relationship, GraphData

logger = logging.getLogger(__name__)


class LineageGraphBuilder:
    """Builds a lineage graph from scraped model data."""

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
        model_nodes = []
        for model_data in models:
            try:
                model_node = ModelNode(**model_data)
                model_nodes.append(model_node)
            except Exception as e:
                logger.warning(
                    f"Failed to create model node for {model_data.get('model_id')}: {e}"
                )
                continue

        # Convert datasets to DatasetNode objects
        dataset_nodes = []
        if datasets:
            for dataset_data in datasets:
                try:
                    dataset_node = DatasetNode(**dataset_data)
                    dataset_nodes.append(dataset_node)
                except Exception as e:
                    logger.warning(
                        f"Failed to create dataset node for {dataset_data.get('dataset_id')}: {e}"
                    )
                    continue

        # If no datasets provided, infer from relationships
        if not datasets:
            dataset_ids = set()
            for rel in relationships:
                if rel.get("target_type") == "dataset":
                    dataset_ids.add(rel["target"])

            for dataset_id in dataset_ids:
                dataset_node = DatasetNode(dataset_id=dataset_id, tags=[])
                dataset_nodes.append(dataset_node)

        # Convert relationships to Relationship objects
        relationship_objects = []
        for rel_data in relationships:
            try:
                relationship = Relationship(**rel_data)
                relationship_objects.append(relationship)
            except Exception as e:
                logger.warning(f"Failed to create relationship: {e}")
                continue

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
