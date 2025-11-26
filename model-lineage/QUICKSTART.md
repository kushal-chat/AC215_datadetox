# Quick Start Guide

## Prerequisites

1. **HuggingFace Token**: Get your token from https://huggingface.co/settings/tokens
2. **Docker & Docker Compose**: Ensure Docker is running

## Setup

1. **Set Environment Variables** (in your `.env` file or export):
   ```bash
   export HF_TOKEN=your_huggingface_token_here
   ```

2. **Start Neo4j**:
   ```bash
   docker compose up -d neo4j
   ```

   Wait for Neo4j to be healthy (check with `docker compose ps`)

3. **Initialize DVC** (if not already done):
   ```bash
   dvc init
   ```

## Running the Pipeline

### Full Pipeline (Recommended for first run)

```bash
# Scrape, build graph, load to Neo4j, and commit to DVC
docker compose run model-lineage-scraper uv run python lineage_scraper.py --full
```

### Step-by-Step

```bash
# 1. Scrape models (limit to 100 for testing)
docker compose run model-lineage-scraper uv run python lineage_scraper.py --scrape --limit 100

# 2. Build graph from scraped data
docker compose run model-lineage-scraper uv run python lineage_scraper.py --build-graph

# 3. Load graph to Neo4j (clearing existing data)
docker compose run model-lineage-scraper uv run python lineage_scraper.py --load-neo4j --clear

# 4. Commit to version control
docker compose run model-lineage-scraper uv run python lineage_scraper.py --commit
```

## Accessing Neo4j

- **Web UI**: http://localhost:7474
  - Username: `neo4j`
  - Password: `password` (change in production!)

- **Example Cypher Query**:
  ```cypher
  // Get all models
  MATCH (m:Model) RETURN m LIMIT 10

  // Get model lineage
  MATCH path = (m:Model {model_id: "bert-base-uncased"})-[*1..3]-(related)
  RETURN path

  // Get statistics
  MATCH (m:Model) RETURN count(m) as model_count
  MATCH ()-[r]->() RETURN count(r) as relationship_count
  ```

## Data Storage

- **Raw Data**: `data/model-lineage/raw/`
  - Models: `data/model-lineage/raw/models/models_*.json`
  - Relationships: `data/model-lineage/raw/relationships/relationships_*.json`

- **DVC Tracking**: All data files have corresponding `.dvc` pointer files tracked in Git

## Troubleshooting

1. **Neo4j Connection Error**:
   - Ensure Neo4j is running: `docker compose ps neo4j`
   - Check health: `docker compose logs neo4j`

2. **DVC Errors**:
   - Ensure DVC is initialized: `dvc init`
   - Check if you're in a Git repository (DVC requires Git)

3. **Rate Limiting**:
   - Adjust `RATE_LIMIT_DELAY` in `config/settings.py` if hitting HuggingFace rate limits

4. **Memory Issues**:
   - Use `--limit` flag to scrape fewer models
   - Increase Neo4j memory in `docker-compose.yml` if needed
