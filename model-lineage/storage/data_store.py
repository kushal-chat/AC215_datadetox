"""Data storage with DVC versioning for lineage scraping pipeline."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class DVCDataStore:
    """Store scraped data with DVC versioning."""

    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            base_path = settings.BASE_DATA_PATH

        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"

        # Create directories
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

        # Ensure DVC is initialized
        self._ensure_dvc_init()

    def _find_project_root(self) -> Optional[Path]:
        """Find the project root directory (where .git and .dvc are)."""
        # Check if we're in Docker with mounted workspace
        workspace_path = Path("/workspace")
        if workspace_path.exists() and (workspace_path / ".git").exists():
            return workspace_path

        # Try to find project root by walking up from current directory
        current = Path.cwd()
        while current != current.parent:
            if (current / ".git").exists() or (current / ".dvc").exists():
                return current
            current = current.parent

        return None

    def _ensure_dvc_init(self):
        """Ensure DVC is initialized in the project root."""
        project_root = self._find_project_root()
        if project_root is None:
            logger.debug(
                "Could not find project root for DVC init. Will skip DVC operations."
            )
            return

        dvc_dir = project_root / ".dvc"

        if not dvc_dir.exists():
            logger.info("Initializing DVC...")
            try:
                subprocess.run(
                    ["dvc", "init"], cwd=project_root, check=True, capture_output=True
                )
                logger.info("DVC initialized")
            except subprocess.CalledProcessError as e:
                logger.warning(f"DVC init failed (may already be initialized): {e}")

    def _save_data(
        self,
        data: List[Dict[str, Any]],
        data_type: str,
        subdirectory: str,
        timestamp: Optional[str] = None,
    ) -> str:
        """
        Generic method to save data with DVC tracking.

        Args:
            data: Data to save
            data_type: Type of data (e.g., "models", "datasets")
            subdirectory: Subdirectory name (e.g., "models", "datasets")
            timestamp: Optional timestamp string

        Returns:
            Path to saved file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        filename = f"{data_type}_{timestamp}.json"
        filepath = self.raw_path / subdirectory / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self._dvc_add(filepath)

        logger.info(f"Saved {len(data)} {data_type} to {filepath}")
        return str(filepath)

    def save_scraped_models(
        self, models: List[Dict[str, Any]], timestamp: Optional[str] = None
    ) -> str:
        """Save scraped model data with DVC tracking."""
        return self._save_data(models, "models", "models", timestamp)

    def save_scraped_datasets(
        self, datasets: List[Dict[str, Any]], timestamp: Optional[str] = None
    ) -> str:
        """Save scraped dataset data with DVC tracking."""
        return self._save_data(datasets, "datasets", "datasets", timestamp)

    def save_relationships(
        self, relationships: List[Dict[str, Any]], timestamp: Optional[str] = None
    ) -> str:
        """
        Save relationships with DVC tracking.
        Automatically filters to only include: finetuned, adapters, merges, quantizations, trained_on
        """
        # Filter relationships to only include allowed types
        relationships = self.filter_relationships(relationships)
        return self._save_data(
            relationships, "relationships", "relationships", timestamp
        )

    def filter_relationships(
        self, relationships: List[Dict[str, Any]], allowed_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter relationships to only include allowed relationship types.

        Args:
            relationships: List of relationship dictionaries
            allowed_types: List of allowed relationship types.
                          Default: ["finetuned", "adapters", "merges", "quantizations"]

        Returns:
            Filtered list of relationships
        """
        if allowed_types is None:
            allowed_types = [
                "finetuned",
                "adapters",
                "merges",
                "quantizations",
                "trained_on",
            ]

        filtered = [
            rel
            for rel in relationships
            if rel.get("relationship_type") in allowed_types
        ]

        logger.info(
            f"Filtered {len(relationships)} relationships to {len(filtered)} "
            f"(keeping only: {', '.join(allowed_types)})"
        )
        return filtered

    def save_metadata(
        self, metadata: Dict[str, Any], timestamp: Optional[str] = None
    ) -> str:
        """Save scraping metadata."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        metadata["timestamp"] = timestamp
        filename = f"scrape_metadata_{timestamp}.json"
        filepath = self.raw_path / "metadata" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        self._dvc_add(filepath)
        return str(filepath)

    def _load_latest_file(
        self, subdirectory: str, pattern: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generic method to load the most recent file matching a pattern.

        Args:
            subdirectory: Subdirectory to search in
            pattern: File pattern to match (e.g., "models_*.json")

        Returns:
            Loaded data or None if no file found
        """
        directory = self.raw_path / subdirectory
        if not directory.exists():
            return None

        files = sorted(directory.glob(pattern), reverse=True)
        if not files:
            return None

        latest_file = files[0]
        with open(latest_file, "r") as f:
            return json.load(f)

    def load_latest_models(self) -> Optional[List[Dict[str, Any]]]:
        """Load the most recent models file."""
        return self._load_latest_file("models", "models_*.json")

    def load_latest_relationships(self) -> Optional[List[Dict[str, Any]]]:
        """Load the most recent relationships file."""
        return self._load_latest_file("relationships", "relationships_*.json")

    def _dvc_add(self, filepath: Path):
        """Add file to DVC tracking."""
        try:
            # Resolve to absolute path first
            filepath = Path(filepath).resolve()

            # Find project root
            project_root = self._find_project_root()
            if project_root is None:
                logger.warning("Could not find Git repository. Skipping DVC tracking.")
                return

            # Get relative path from project root
            try:
                rel_path = filepath.relative_to(project_root)
            except ValueError:
                # If filepath is not relative to project_root, handle Docker paths
                # In Docker, data is mounted at /app/data/model-lineage
                # but project root is at /workspace
                filepath_str = str(filepath)

                # Check if we're in Docker (/app) and project root is at /workspace
                if filepath_str.startswith("/app/data/"):
                    # Convert /app/data/model-lineage/... to data/model-lineage/...
                    rel_path = Path(filepath_str[5:])  # Remove "/app"
                elif filepath_str.startswith("/app/"):
                    # For other /app paths, try to map to workspace
                    # /app/... should map to workspace/model-lineage/...
                    rel_path = Path("model-lineage") / filepath_str[5:]
                elif "data/model-lineage" in filepath_str:
                    # Extract the data/model-lineage part
                    idx = filepath_str.find("data/model-lineage")
                    rel_path = Path(filepath_str[idx:])
                else:
                    logger.warning(f"Could not determine relative path for {filepath}")
                    return  # Skip DVC add if we can't determine relative path

            # Verify the file exists at the relative path in project root
            full_path_in_project = project_root / rel_path
            if not full_path_in_project.exists():
                logger.warning(
                    f"File {rel_path} does not exist in project root {project_root}. Skipping DVC add."
                )
                return

            # Run dvc add from project root
            subprocess.run(
                ["dvc", "add", str(rel_path)],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f"Added {rel_path} to DVC")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add {filepath} to DVC: {e.stderr}")
            # Don't raise - allow pipeline to continue even if DVC fails
            logger.warning("Continuing without DVC tracking")
        except Exception as e:
            logger.error(f"Error in _dvc_add for {filepath}: {e}")
            logger.warning("Continuing without DVC tracking")

    def commit_version(self, message: Optional[str] = None):
        """Commit current version to DVC and Git."""
        if message is None:
            message = f"Lineage data update: {datetime.now().isoformat()}"

        try:
            # Find project root
            project_root = self._find_project_root()
            if project_root is None:
                logger.warning(
                    "Could not find Git repository. Skipping version commit."
                )
                return

            # DVC commit
            subprocess.run(
                ["dvc", "commit"], cwd=project_root, check=True, capture_output=True
            )

            # Git commit (if in git repo)
            try:
                subprocess.run(
                    ["git", "add", "data/model-lineage/**/*.dvc", ".dvc"],
                    cwd=project_root,
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", message],
                    cwd=project_root,
                    check=True,
                    capture_output=True,
                )
                logger.info(f"Committed version: {message}")
            except subprocess.CalledProcessError:
                logger.warning(
                    "Git commit failed (may not be in git repo or no changes)"
                )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit version: {e}")
            # Don't raise - allow pipeline to continue

    def cleanup_old_files(self, keep_latest: int, file_type: str = "models"):
        """
        Clean up old files, keeping only the N most recent.

        Args:
            keep_latest: Number of most recent files to keep
            file_type: Type of files to clean ("models", "relationships", or "metadata")
        """
        if keep_latest <= 0:
            logger.warning(
                f"keep_latest must be > 0, got {keep_latest}. Skipping cleanup."
            )
            return

        # Determine directory and file pattern based on file_type
        if file_type == "models":
            directory = self.raw_path / "models"
            pattern = "models_*.json"
        elif file_type == "datasets":
            directory = self.raw_path / "datasets"
            pattern = "datasets_*.json"
        elif file_type == "relationships":
            directory = self.raw_path / "relationships"
            pattern = "relationships_*.json"
        elif file_type == "metadata":
            directory = self.raw_path / "metadata"
            pattern = "scrape_metadata_*.json"
        else:
            logger.error(f"Unknown file_type: {file_type}")
            return

        if not directory.exists():
            logger.debug(f"Directory {directory} does not exist. Nothing to clean.")
            return

        # Find all matching files
        files = list(directory.glob(pattern))

        if len(files) <= keep_latest:
            logger.debug(
                f"Only {len(files)} files found, keeping all (limit: {keep_latest})"
            )
            return

        # Sort by modification time (most recent first)
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Files to keep (most recent N)
        files_to_delete = files[keep_latest:]

        deleted_count = 0
        for filepath in files_to_delete:
            try:
                # Delete the data file
                filepath.unlink()
                deleted_count += 1
                logger.debug(f"Deleted old file: {filepath.name}")

                # Also delete corresponding .dvc file if it exists
                dvc_file = filepath.with_suffix(filepath.suffix + ".dvc")
                if dvc_file.exists():
                    dvc_file.unlink()
                    logger.debug(f"Deleted old DVC file: {dvc_file.name}")
            except Exception as e:
                logger.error(f"Failed to delete {filepath}: {e}")

        if deleted_count > 0:
            logger.info(
                f"Cleaned up {deleted_count} old {file_type} file(s), kept {keep_latest} most recent"
            )
        else:
            logger.debug(f"No files to clean up for {file_type}")
