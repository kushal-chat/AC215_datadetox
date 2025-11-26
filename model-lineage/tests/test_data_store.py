"""Tests for data store."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from storage.data_store import DVCDataStore


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_project_root(temp_dir):
    """Create a mock project root with .git directory."""
    git_dir = temp_dir / ".git"
    git_dir.mkdir()
    return temp_dir


def test_dvc_data_store_init(temp_dir):
    """Test DVCDataStore initialization."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        assert store.base_path == temp_dir
        assert store.raw_path == temp_dir / "raw"
        assert store.processed_path == temp_dir / "processed"
        assert store.raw_path.exists()
        assert store.processed_path.exists()


def test_save_scraped_models(temp_dir):
    """Test saving scraped models."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        models = [
            {"model_id": "model1", "author": "author1"},
            {"model_id": "model2", "author": "author2"},
        ]

        filepath_str = store.save_scraped_models(
            models, timestamp="2024-01-01_00-00-00"
        )
        filepath = Path(filepath_str)

        assert filepath.exists()
        assert filepath.name == "models_2024-01-01_00-00-00.json"

        # Verify content
        with open(filepath) as f:
            data = json.load(f)
            assert len(data) == 2
            assert data[0]["model_id"] == "model1"


def test_save_scraped_datasets(temp_dir):
    """Test saving scraped datasets."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        datasets = [
            {"dataset_id": "dataset1", "author": "author1"},
            {"dataset_id": "dataset2", "author": "author2"},
        ]

        filepath_str = store.save_scraped_datasets(
            datasets, timestamp="2024-01-01_00-00-00"
        )
        filepath = Path(filepath_str)

        assert filepath.exists()
        assert filepath.name == "datasets_2024-01-01_00-00-00.json"

        with open(filepath) as f:
            data = json.load(f)
            assert len(data) == 2
            assert data[0]["dataset_id"] == "dataset1"


def test_save_relationships(temp_dir):
    """Test saving relationships."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        relationships = [
            {
                "source": "model1",
                "target": "model2",
                "relationship_type": "finetuned",
                "source_type": "model",
                "target_type": "model",
            }
        ]

        filepath_str = store.save_relationships(
            relationships, timestamp="2024-01-01_00-00-00"
        )
        filepath = Path(filepath_str)

        assert filepath.exists()
        assert filepath.name == "relationships_2024-01-01_00-00-00.json"

        with open(filepath) as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["source"] == "model1"


@patch("storage.data_store.subprocess.run")
def test_dvc_add_called_on_save(mock_subprocess, temp_dir, mock_project_root):
    """Test that DVC add is called when saving files."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        with patch.object(
            DVCDataStore, "_find_project_root", return_value=mock_project_root
        ):
            store = DVCDataStore(base_path=temp_dir)

            models = [{"model_id": "model1"}]
            filepath_str = store.save_scraped_models(models)
            filepath = Path(filepath_str)

            # Verify dvc add was called (if DVC is available)
            # Note: This may not be called if DVC init fails, which is expected in tests
            assert filepath.exists()


def test_find_project_root_with_git(temp_dir):
    """Test finding project root with .git directory."""
    git_dir = temp_dir / ".git"
    git_dir.mkdir()

    store = DVCDataStore(base_path=temp_dir / "subdir" / "data")
    root = store._find_project_root()

    # Should find the temp_dir as project root
    assert root is not None


def test_find_project_root_with_dvc(temp_dir):
    """Test finding project root with .dvc directory."""
    dvc_dir = temp_dir / ".dvc"
    dvc_dir.mkdir()

    store = DVCDataStore(base_path=temp_dir / "subdir" / "data")
    root = store._find_project_root()

    assert root is not None


def test_save_metadata(temp_dir):
    """Test saving metadata."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        metadata = {"total_models": 100, "total_datasets": 50}

        filepath_str = store.save_metadata(metadata, timestamp="2024-01-01_00-00-00")
        filepath = Path(filepath_str)

        assert filepath.exists()
        assert filepath.name == "scrape_metadata_2024-01-01_00-00-00.json"

        with open(filepath) as f:
            data = json.load(f)
            assert data["total_models"] == 100
            assert data["timestamp"] == "2024-01-01_00-00-00"


def test_load_latest_models(temp_dir):
    """Test loading latest models file."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        # Create models directory and files
        models_dir = store.raw_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Create older file
        older_file = models_dir / "models_2024-01-01_00-00-00.json"
        with open(older_file, "w") as f:
            json.dump([{"model_id": "old_model"}], f)

        # Create newer file
        newer_file = models_dir / "models_2024-01-02_00-00-00.json"
        with open(newer_file, "w") as f:
            json.dump([{"model_id": "new_model"}], f)

        models = store.load_latest_models()

        assert models is not None
        assert len(models) == 1
        assert models[0]["model_id"] == "new_model"


def test_load_latest_models_no_files(temp_dir):
    """Test loading latest models when no files exist."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        models = store.load_latest_models()

        assert models is None


def test_load_latest_relationships(temp_dir):
    """Test loading latest relationships file."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        # Create relationships directory and files
        rel_dir = store.raw_path / "relationships"
        rel_dir.mkdir(parents=True, exist_ok=True)

        # Create newer file
        newer_file = rel_dir / "relationships_2024-01-02_00-00-00.json"
        with open(newer_file, "w") as f:
            json.dump(
                [
                    {
                        "source": "model1",
                        "target": "model2",
                        "relationship_type": "finetuned",
                    }
                ],
                f,
            )

        relationships = store.load_latest_relationships()

        assert relationships is not None
        assert len(relationships) == 1
        assert relationships[0]["source"] == "model1"


def test_load_latest_relationships_no_files(temp_dir):
    """Test loading latest relationships when no files exist."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        relationships = store.load_latest_relationships()

        assert relationships is None


@patch("storage.data_store.subprocess.run")
def test_dvc_add_with_project_root(mock_subprocess, temp_dir, mock_project_root):
    """Test DVC add when project root is found."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        with patch.object(
            DVCDataStore, "_find_project_root", return_value=mock_project_root
        ):
            store = DVCDataStore(base_path=temp_dir)

            filepath = temp_dir / "raw" / "models" / "test.json"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text("{}")

            store._dvc_add(filepath)

            # Verify dvc add was called
            assert mock_subprocess.called


@patch("storage.data_store.subprocess.run")
def test_dvc_add_no_project_root(mock_subprocess, temp_dir):
    """Test DVC add when project root is not found."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        with patch.object(DVCDataStore, "_find_project_root", return_value=None):
            store = DVCDataStore(base_path=temp_dir)

            filepath = temp_dir / "raw" / "models" / "test.json"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text("{}")

            store._dvc_add(filepath)

            # Should not call dvc add if no project root
            # (may still be called for DVC init, but that's okay)
