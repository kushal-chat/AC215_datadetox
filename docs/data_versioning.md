# Data Versioning Strategy

## Chosen Method: DVC (Data Version Control)

We use **DVC** (diff-based versioning) to version all scraped model lineage data, including models, datasets, relationships, and metadata.

## Justification

**Why DVC over snapshot-based methods:**

1. **Storage efficiency**: Scraped JSON files are large (150KB+ per file). DVC uses content-addressable storage with deduplication, avoiding full copies of unchanged data.

2. **Git integration**: Only small `.dvc` pointer files are committed to Git. Actual data files are stored separately, keeping the repository lightweight.

3. **Incremental updates**: New scraping runs create timestamped files. DVC tracks only changes, not full snapshots.

4. **Network efficiency**: Only changed content is transferred, not entire files.

## Data Characteristics

- **Static once scraped**: Scraped data files don't change after creation; they're append-only snapshots.
- **Periodic updates**: New scraping runs create new timestamped files (e.g., `models_2025-11-23_03-36-48.json`).
- **Large files**: JSON files containing model/dataset metadata can be substantial.
- **Multiple versions**: Historical versions are maintained for reproducibility.

DVC fits this use case because it handles large, relatively static files efficiently while maintaining full version history.

## Reproducibility

Each scraping run is versioned with:
- Timestamped data files
- `.dvc` pointer files tracked in Git
- Commit messages linking code and data versions

This enables exact data retrieval for any historical scraping run.

## Version History

Version history is maintained through:
- **Git commits**: `.dvc` pointer files are committed to Git with descriptive messages
- **Timestamped files**: Each scraping run creates files with timestamps (e.g., `models_2025-11-23_03-36-48.json`)
- **DVC cache**: Historical versions are stored in DVC's cache or remote storage

View version history:
```bash
git log --oneline data/model-lineage/raw/**/*.dvc
```

## Data Retrieval

### Pull data for a specific commit:
```bash
# Checkout the commit
git checkout <commit-hash>

# Pull the corresponding data files
dvc pull
```

### Pull latest data:
```bash
dvc pull
```

### Push data to remote storage (if configured):
```bash
dvc push
```

### Manual workflow:
```bash
# After scraping, data is automatically added to DVC
# To commit version:
python lineage_scraper.py --commit --message "Scraping run: 2025-11-23"

# Or manually:
dvc commit
git add data/model-lineage/**/*.dvc .dvc
git commit -m "Update lineage data"
```

## Implementation

The `DVCDataStore` class (`model-lineage/storage/data_store.py`) handles:
- Automatic DVC initialization
- File tracking after save operations
- Version commits to DVC and Git
- Cleanup of old local files while maintaining version history
