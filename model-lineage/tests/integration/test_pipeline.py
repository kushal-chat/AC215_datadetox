"""Integration tests for the model-lineage pipeline."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from lineage_scraper import scrape_models, build_graph, load_to_neo4j
from storage.data_store import DVCDataStore


@pytest.fixture
def temp_data_store():
    """Create a temporary data store for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DVCDataStore(base_path=Path(tmpdir))
        yield store


@pytest.fixture
def mock_scraper():
    """Mock the HuggingFace scraper."""
    with patch("lineage_scraper.HuggingFaceScraper") as mock_scraper_class:
        mock_scraper = MagicMock()
        mock_scraper_class.return_value = mock_scraper

        # Mock scraped data
        mock_scraper.scrape_all_models.return_value = (
            [
                {
                    "model_id": "test/model1",
                    "author": "test",
                    "downloads": 1000,
                    "likes": 50,
                    "tags": ["nlp"],
                    "pipeline_tag": "text-generation",
                    "library_name": "transformers",
                    "private": False,
                    "url": "https://huggingface.co/test/model1",
                }
            ],
            [
                {
                    "dataset_id": "test/dataset1",
                    "author": "test",
                    "downloads": 500,
                    "tags": ["nlp"],
                }
            ],
            [
                {
                    "source": "test/model1",
                    "target": "test/dataset1",
                    "relationship_type": "trained_on",
                    "source_type": "model",
                    "target_type": "dataset",
                }
            ],
        )
        mock_scraper.scrape_datasets.return_value = ([], [])
        yield mock_scraper


def test_scrape_models_integration(temp_data_store, mock_scraper):
    """Test the scraping stage of the pipeline."""
    models_path, datasets_path, rels_path, metadata_path = scrape_models(
        temp_data_store, limit=1
    )

    # Verify files were created
    assert Path(models_path).exists()
    assert Path(datasets_path).exists()
    assert Path(rels_path).exists()
    assert Path(metadata_path).exists()

    # Verify content
    import json

    with open(models_path) as f:
        models = json.load(f)
        assert len(models) == 1
        assert models[0]["model_id"] == "test/model1"


def test_build_graph_integration(temp_data_store, mock_scraper):
    """Test the graph building stage."""
    # First scrape
    scrape_models(temp_data_store, limit=1)

    # Then build graph
    graph_data = build_graph(temp_data_store)

    assert graph_data is not None
    assert len(graph_data.models) > 0
    assert len(graph_data.datasets) > 0
    assert len(graph_data.relationships) > 0


@patch("lineage_scraper.Neo4jClient")
def test_load_to_neo4j_integration(
    mock_neo4j_client_class, temp_data_store, mock_scraper
):
    """Test loading graph to Neo4j."""
    # Mock Neo4j client
    mock_client = MagicMock()
    mock_client.get_statistics.return_value = {
        "model_count": 1,
        "dataset_count": 1,
        "relationship_count": 1,
    }
    mock_neo4j_client_class.return_value = mock_client

    # Scrape and build
    scrape_models(temp_data_store, limit=1)
    graph_data = build_graph(temp_data_store)

    # Load to Neo4j
    load_to_neo4j(graph_data, clear_existing=True)

    # Verify Neo4j client was called
    mock_client.load_graph.assert_called_once_with(graph_data)
    mock_client.clear_database.assert_called_once()
    mock_client.get_statistics.assert_called_once()


def test_full_pipeline_integration(temp_data_store, mock_scraper):
    """Test the complete pipeline from scraping to graph building."""
    # Stage 1: Scrape
    models_path, datasets_path, rels_path, metadata_path = scrape_models(
        temp_data_store, limit=1
    )

    # Stage 2: Build graph
    graph_data = build_graph(temp_data_store)

    # Verify graph structure
    assert graph_data is not None
    assert len(graph_data.models) == 1
    assert len(graph_data.datasets) == 1
    assert len(graph_data.relationships) == 1

    # Verify relationships are correct
    rel = graph_data.relationships[0]
    assert rel.source == "test/model1"
    assert rel.target == "test/dataset1"
    assert rel.relationship_type == "trained_on"


@patch("lineage_scraper.Neo4jClient")
def test_pipeline_with_neo4j_clear(
    mock_neo4j_client_class, temp_data_store, mock_scraper
):
    """Test pipeline with Neo4j clear option."""
    mock_client = MagicMock()
    mock_client.get_statistics.return_value = {
        "model_count": 1,
        "dataset_count": 1,
        "relationship_count": 1,
    }
    mock_neo4j_client_class.return_value = mock_client

    # Scrape and build
    scrape_models(temp_data_store, limit=1)
    graph_data = build_graph(temp_data_store)

    # Load with clear_existing=True
    load_to_neo4j(graph_data, clear_existing=True)
    assert mock_client.clear_database.called

    # Load with clear_existing=False
    load_to_neo4j(graph_data, clear_existing=False)
    # Should not call clear again (only once from previous call)
    assert mock_client.clear_database.call_count == 1
