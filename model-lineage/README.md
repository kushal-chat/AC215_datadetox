# Model Lineage Scraper

A pipeline for scraping HuggingFace model trees and building model lineage graphs stored in Neo4j.

## Overview

This pipeline:
1. Scrapes model information from HuggingFace Hub
2. Extracts model relationships (base models, fine-tuned versions, etc.)
3. Builds a lineage graph
4. Stores the graph in Neo4j
5. Versions all data using DVC (See instructions for data versioning at `docs/data_versioning.md`)

## Setup

### Prerequisites

- Docker and Docker Compose
- HuggingFace token (set `HF_TOKEN` environment variable)
- Neo4j running (via docker-compose)

### Environment Variables

```bash
HF_TOKEN=your_huggingface_token
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## Usage

### Run Full Pipeline

```bash
docker compose run model-lineage-scraper uv run python lineage_scraper.py --full
```

### Run Individual Stages

```bash
# Scrape HuggingFace models
docker compose run model-lineage-scraper uv run python lineage_scraper.py --scrape

# Build lineage graph
docker compose run model-lineage-scraper uv run python lineage_scraper.py --build-graph

# Load graph to Neo4j
docker compose run model-lineage-scraper uv run python lineage_scraper.py --load-neo4j

# Commit to DVC
docker compose run model-lineage-scraper uv run python lineage_scraper.py --commit
```

## Data Storage

- Raw scraped data: `data/model-lineage/raw/`
- Processed graph data: `data/model-lineage/processed/`
- All data is versioned with DVC (pointer files tracked in Git)

## Neo4j Access

- HTTP: http://localhost:7474
- Bolt: bolt://localhost:7687
- Default credentials: neo4j/password (change in production!)
