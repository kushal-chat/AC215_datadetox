"""
Main orchestration script for model lineage scraping.

This script coordinates the full pipeline:
1. Scrape HuggingFace models
2. Build lineage graph
3. Load graph to Neo4j
4. Version data with DVC
"""

import argparse
import logging
from datetime import datetime
from typing import Optional

from scrapers.huggingface_scraper import HuggingFaceScraper
from graph.builder import LineageGraphBuilder
from graph.neo4j_client import Neo4jClient
from storage.data_store import DVCDataStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _log_section(title: str):
    """Log a formatted section header."""
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def scrape_models(
    data_store: DVCDataStore, limit: int = None, keep_latest: Optional[int] = None
) -> tuple[str, str, str, str]:
    """Stage 1: Scrape models and datasets from HuggingFace."""
    _log_section("Stage 1: Scraping HuggingFace models and datasets")

    scraper = HuggingFaceScraper()
    models, datasets, relationships = scraper.scrape_all_models(limit=limit)

    # Collect unique dataset IDs for scraping
    dataset_ids = list(set(d["dataset_id"] for d in datasets))

    # Scrape datasets to get full info and additional relationships
    if dataset_ids:
        logger.info(f"Scraping {len(dataset_ids)} datasets...")
        scraped_datasets, dataset_rels = scraper.scrape_datasets(dataset_ids)

        # Merge scraped datasets with ones found from model tags
        dataset_dict = {d["dataset_id"]: d for d in datasets}
        for dataset in scraped_datasets:
            if dataset["dataset_id"] not in dataset_dict:
                datasets.append(dataset)
            else:
                # Update with more complete info
                dataset_dict[dataset["dataset_id"]].update(dataset)

        relationships.extend(dataset_rels)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save with DVC
    models_path = data_store.save_scraped_models(models, timestamp)
    datasets_path = data_store.save_scraped_datasets(datasets, timestamp)
    relationships_path = data_store.save_relationships(relationships, timestamp)
    metadata_path = data_store.save_metadata(
        {
            "total_models": len(models),
            "total_datasets": len(datasets),
            "total_relationships": len(relationships),
            "scrape_timestamp": timestamp,
        },
        timestamp,
    )

    # Clean up old files if keep_latest is specified
    if keep_latest is not None:
        data_store.cleanup_old_files(keep_latest, "models")
        data_store.cleanup_old_files(keep_latest, "datasets")
        data_store.cleanup_old_files(keep_latest, "relationships")
        data_store.cleanup_old_files(keep_latest, "metadata")

    logger.info(
        f"✓ Scraped {len(models)} models, {len(datasets)} datasets, and {len(relationships)} relationships"
    )
    return models_path, datasets_path, relationships_path, metadata_path


def build_graph(data_store: DVCDataStore) -> tuple:
    """Stage 2: Build lineage graph from scraped data."""
    _log_section("Stage 2: Building lineage graph")

    # Load latest data
    models = data_store.load_latest_models()
    relationships = data_store.load_latest_relationships()

    if not models:
        raise ValueError("No models found. Run --scrape first.")

    if not relationships:
        logger.warning("No relationships found. Graph will only contain nodes.")
        relationships = []

    # Build graph
    builder = LineageGraphBuilder()
    graph_data = builder.build_from_data(models, relationships)

    logger.info(
        f"✓ Built graph with {len(graph_data.models)} models, "
        f"{len(graph_data.datasets)} datasets, "
        f"and {len(graph_data.relationships)} relationships"
    )

    return graph_data


def load_to_neo4j(graph_data, clear_existing: bool = False):
    """Stage 3: Load graph to Neo4j."""
    _log_section("Stage 3: Loading graph to Neo4j")

    neo4j = Neo4jClient()

    try:
        if clear_existing:
            logger.info("Clearing existing Neo4j data...")
            neo4j.clear_database()

        neo4j.load_graph(graph_data)

        # Get statistics
        stats = neo4j.get_statistics()
        logger.info("✓ Graph loaded successfully:")
        logger.info(f"  - Models: {stats.get('model_count', 0)}")
        logger.info(f"  - Datasets: {stats.get('dataset_count', 0)}")
        logger.info(f"  - Relationships: {stats.get('relationship_count', 0)}")

    finally:
        neo4j.close()


def commit_data(data_store: DVCDataStore, message: str = None):
    """Stage 4: Commit data to DVC and Git."""
    _log_section("Stage 4: Committing to version control")

    if message is None:
        message = f"Lineage data update: {datetime.now().isoformat()}"

    data_store.commit_version(message)
    logger.info("✓ Data committed to version control")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape HuggingFace model lineage and build graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python lineage_scraper.py --full

  # Scrape only (limit to 100 models for testing)
  python lineage_scraper.py --scrape --limit 100

  # Build graph from existing data
  python lineage_scraper.py --build-graph

  # Load to Neo4j (clearing existing data)
  python lineage_scraper.py --load-neo4j --clear
        """,
    )

    parser.add_argument("--scrape", action="store_true", help="Run scraping stage")
    parser.add_argument(
        "--build-graph",
        action="store_true",
        help="Build lineage graph from scraped data",
    )
    parser.add_argument("--load-neo4j", action="store_true", help="Load graph to Neo4j")
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing Neo4j data before loading"
    )
    parser.add_argument("--commit", action="store_true", help="Commit to DVC and Git")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline (scrape -> build -> load -> commit)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of models to scrape (for testing)",
    )
    parser.add_argument(
        "--keep-latest",
        type=int,
        default=None,
        help="Keep only the N most recent files, delete older ones",
    )
    parser.add_argument(
        "--message", type=str, default=None, help="Commit message for version control"
    )

    args = parser.parse_args()

    # If --full, set all stages
    if args.full:
        args.scrape = True
        args.build_graph = True
        args.load_neo4j = True
        args.commit = True

    if not any([args.scrape, args.build_graph, args.load_neo4j, args.commit]):
        parser.print_help()
        return

    try:
        data_store = DVCDataStore()

        # Stage 1: Scrape
        if args.scrape:
            scrape_models(data_store, limit=args.limit, keep_latest=args.keep_latest)

        # Stage 2: Build graph
        graph_data = None
        if args.build_graph or args.load_neo4j:
            graph_data = build_graph(data_store)

        # Stage 3: Load to Neo4j
        if args.load_neo4j:
            if graph_data is None:
                graph_data = build_graph(data_store)
            load_to_neo4j(graph_data, clear_existing=args.clear)

        # Stage 4: Commit
        if args.commit:
            commit_data(data_store, message=args.message)

        _log_section("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
