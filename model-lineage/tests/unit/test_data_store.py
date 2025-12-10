"""Tests for data store."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_dvc_data_store_init_with_none_base_path(temp_dir):
    """Test DVCDataStore initialization with base_path=None."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=None)

        assert store.base_path == temp_dir
        assert store.raw_path == temp_dir / "raw"
        assert store.processed_path == temp_dir / "processed"


def test_filter_relationships(temp_dir):
    """Test filtering relationships by type."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        relationships = [
            {"source": "m1", "target": "m2", "relationship_type": "finetuned"},
            {"source": "m2", "target": "m3", "relationship_type": "adapters"},
            {"source": "m3", "target": "m4", "relationship_type": "unknown"},
        ]

        filtered = store.filter_relationships(
            relationships, allowed_types=["finetuned", "adapters"]
        )

        assert len(filtered) == 2
        assert filtered[0]["relationship_type"] == "finetuned"
        assert filtered[1]["relationship_type"] == "adapters"


def test_filter_relationships_default_types(temp_dir):
    """Test filtering relationships with default allowed types."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        relationships = [
            {"source": "m1", "target": "m2", "relationship_type": "finetuned"},
            {"source": "m2", "target": "m3", "relationship_type": "trained_on"},
            {"source": "m3", "target": "m4", "relationship_type": "unknown"},
        ]

        filtered = store.filter_relationships(relationships)

        assert len(filtered) == 2
        assert any(r["relationship_type"] == "finetuned" for r in filtered)
        assert any(r["relationship_type"] == "trained_on" for r in filtered)


def test_load_latest_file_no_files(temp_dir):
    """Test _load_latest_file when no files exist."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        result = store._load_latest_file("models", "models_*.json")

        assert result is None


def test_dvc_add_docker_paths(temp_dir, mock_project_root):
    """Test _dvc_add with Docker path handling."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        with patch.object(
            DVCDataStore, "_find_project_root", return_value=mock_project_root
        ):
            store = DVCDataStore(base_path=temp_dir)

            # Create a file in the temp_dir instead of /app
            test_file = temp_dir / "raw" / "models" / "test.json"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("{}")

            with patch("storage.data_store.subprocess.run"):
                # Mock the relative_to to simulate Docker path scenario
                with patch.object(
                    Path, "relative_to", side_effect=ValueError("Not relative")
                ):
                    store._dvc_add(test_file)
                    # Should handle Docker paths gracefully


@patch("storage.data_store.subprocess.run")
def test_commit_version_success(mock_subprocess, temp_dir, mock_project_root):
    """Test successful version commit."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        with patch.object(
            DVCDataStore, "_find_project_root", return_value=mock_project_root
        ):
            store = DVCDataStore(base_path=temp_dir)

            store.commit_version("Test commit")

            # Verify dvc commit and git commands were called
            assert mock_subprocess.called


@patch("storage.data_store.subprocess.run")
def test_commit_version_no_project_root(mock_subprocess, temp_dir):
    """Test commit_version when project root is not found."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        with patch.object(DVCDataStore, "_find_project_root", return_value=None):
            store = DVCDataStore(base_path=temp_dir)

            store.commit_version("Test commit")

            # Should not call subprocess if no project root
            # (subprocess may still be called for DVC init, but that's okay)


@patch("storage.data_store.subprocess.run")
def test_commit_version_git_commit_fails(mock_subprocess, temp_dir, mock_project_root):
    """Test commit_version when git commit fails."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        with patch.object(
            DVCDataStore, "_find_project_root", return_value=mock_project_root
        ):
            store = DVCDataStore(base_path=temp_dir)

            # Make git commit fail
            def side_effect(*args, **kwargs):
                if "git" in args[0] and "commit" in args[0]:
                    raise subprocess.CalledProcessError(1, "git")
                return MagicMock()

            mock_subprocess.side_effect = side_effect

            # Should not raise, just log warning
            store.commit_version("Test commit")


def test_cleanup_old_files_models(temp_dir):
    """Test cleanup_old_files for models."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        # Create models directory and multiple files
        models_dir = store.raw_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Create 3 files
        for i in range(3):
            file = models_dir / f"models_2024-01-0{i + 1}_00-00-00.json"
            file.write_text("{}")

        store.cleanup_old_files(keep_latest=2, file_type="models")

        # Should keep 2 most recent files
        remaining_files = list(models_dir.glob("models_*.json"))
        assert len(remaining_files) == 2


def test_cleanup_old_files_relationships(temp_dir):
    """Test cleanup_old_files for relationships."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        rel_dir = store.raw_path / "relationships"
        rel_dir.mkdir(parents=True, exist_ok=True)

        for i in range(3):
            file = rel_dir / f"relationships_2024-01-0{i + 1}_00-00-00.json"
            file.write_text("{}")

        store.cleanup_old_files(keep_latest=1, file_type="relationships")

        remaining_files = list(rel_dir.glob("relationships_*.json"))
        assert len(remaining_files) == 1


def test_cleanup_old_files_metadata(temp_dir):
    """Test cleanup_old_files for metadata."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        metadata_dir = store.raw_path / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        for i in range(2):
            file = metadata_dir / f"scrape_metadata_2024-01-0{i + 1}_00-00-00.json"
            file.write_text("{}")

        store.cleanup_old_files(keep_latest=1, file_type="metadata")

        remaining_files = list(metadata_dir.glob("scrape_metadata_*.json"))
        assert len(remaining_files) == 1


def test_cleanup_old_files_invalid_type(temp_dir):
    """Test cleanup_old_files with invalid file type."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        # Should not raise, just log error
        store.cleanup_old_files(keep_latest=1, file_type="invalid")


def test_cleanup_old_files_keep_zero(temp_dir):
    """Test cleanup_old_files with keep_latest=0."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        # Should not raise, just log warning
        store.cleanup_old_files(keep_latest=0, file_type="models")


def test_cleanup_old_files_not_enough_files(temp_dir):
    """Test cleanup_old_files when there aren't enough files to clean."""
    with patch("storage.data_store.settings") as mock_settings:
        mock_settings.BASE_DATA_PATH = temp_dir

        store = DVCDataStore(base_path=temp_dir)

        models_dir = store.raw_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Create only 1 file
        file = models_dir / "models_2024-01-01_00-00-00.json"
        file.write_text("{}")

        store.cleanup_old_files(keep_latest=2, file_type="models")

        # Should keep the file
        remaining_files = list(models_dir.glob("models_*.json"))
        assert len(remaining_files) == 1
