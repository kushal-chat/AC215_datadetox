# DataDetox Application Design Document

**Team Members**: Kushal Chattopadhyay, Keyu Wang, Terry Zhou
**Group Name**: DataDetox

---

## Executive Summary

DataDetox is an AI-driven system that maps ML model provenance and training data lineage using HuggingFace metadata and a Neo4j graph database. It helps practitioners spot potential risks—such as copyrighted or questionable datasets (e.g., LAION-5B)—by answering natural-language queries and visualizing model dependencies.

**Technology Stack**: React + TypeScript, FastAPI, Python 3.13, OpenAI Agents SDK, Neo4j, Docker

---

## Solution Architecture

### System Overview

DataDetox is organized into three main layers:

**1. User Interface**

* React frontend (Port 3000) with a chat interface and interactive graph visualization
* Communicates with the backend using a REST API

**2. Application Layer**

* FastAPI backend (Port 8000) running the AI agent
* OpenAI Agent with two tools:

  * HuggingFace metadata fetcher
  * Neo4j lineage query tool

**3. Data Pipeline**

* HuggingFace scraper → graph builder → Neo4j loader
* DVC used for versioning and reproducibility

### Data Flow

#### High-Level Flow

```
User Query → Frontend → Backend API → AI Agent
                                       ↓
                     HuggingFace API + Neo4j Query
                                       ↓
                           Synthesized Response
                                       ↓
                Frontend ← { result, graph_data }
                                       ↓
                     Chat Response + Tree Visualization
```

#### Detailed Query Flow

* User sends a query from the chat interface
* Frontend forwards it to the backend
* Backend initializes state and passes it to the agent
* The agent:

  1. Queries HuggingFace for metadata
  2. Queries Neo4j for lineage
  3. Fetches additional details as needed
  4. Produces a final summary and graph data
* Frontend displays the message and renders the lineage tree

![](../assets/img/ms4/query_flow.png)

### Component Interactions (Brief)

1. User enters a query.
2. Frontend sends a POST request to `/flow/search`.
3. Backend sets up context for the agent.
4. Agent fetches metadata from HuggingFace.
5. Agent retrieves lineage data from Neo4j.
6. Agent enriches results and produces the final summary.
7. Backend returns the summary and graph data.
8. Frontend updates the chat and visualization.

---

## Technical Architecture

### Frontend

**Stack**: React 18.3, Vite, TypeScript, TailwindCSS, React-D3-Tree

**Directory Structure**

```
src/
├── components/
│   ├── ui/              // Radix UI components
│   ├── ChatMessage.tsx  // Chat bubbles
│   └── ModelTree.tsx    // Lineage visualization
├── pages/
│   ├── Index.tsx
│   └── Chatbot.tsx
└── hooks/
    └── use-mobile.tsx
```

**Key Features**

* Split-screen layout (chat + graph)
* Live response updates
* D3 tree with zoom/pan
* Markdown rendering

### Backend

**Stack**: FastAPI, Python 3.13, OpenAI Agents SDK, Neo4j Driver

**Directory Structure**

```
backend/
├── main.py
├── routers/
│   └── search/
│       ├── agent.py
│       └── utils/
│           ├── huggingface.py
│           ├── search_neo4j.py
│           └── tool_state.py
└── tests/
```

**Agent Configuration**

```python
instructions = """
1. Query HuggingFace for metadata and IDs.
2. Query Neo4j for connected models/datasets.
3. Fetch extra details from HuggingFace if needed.
4. Summarize findings and highlight risks.
"""

tools = [search_huggingface, search_neo4j]
agent = Agent(name="SearchAgent", model="gpt-5-nano",
              instructions=instructions, tools=tools)
```

**Main Endpoint**

* `POST /flow/search`

  * Input: `{ "query_val": "..." }`
  * Output: `{ "result": "...", "neo4j_data": {...} }`

---

## Data Pipeline Architecture

**Stack**: HuggingFace Hub API, Neo4j, DVC, Pydantic

### Stages

1. **Collection**

   * Scrape models/datasets
   * Extract relationships
   * Store with DVC

2. **Graph Construction**

   * Build nodes and edges
   * Validate structure

3. **Neo4j Loading**

   * Batch import
   * Create indexes

**Graph Schema**

```cypher
(:Model {model_id, downloads, pipeline_tag, tags})
(:Dataset {dataset_id, downloads, is_problematic})

(:Model)-[:TRAINED_ON]->(:Dataset)
(:Model)-[:DERIVED_FROM]->(:Model)
```

---

## User Interface Design

### Chat Interface

* Two-panel layout
* Left: chat, timestamps, typing indicators
* Right: D3 lineage viewer with zoom/pan and tooltips


![Chatbot UI](../assets/img/ms4/chatbot_ui.png)
*Example. Model lineage tree of "Qwen3-4b"*

* **🟡 Yellow node** — the model you asked about
* **🔵 Blue nodes** — related or derived models
* **⚪ White nodes** — datasets tied to those models

---

## Code Organization

### Frontend

* `components/`: UI + shared components
* `pages/`: routing and screens
* `hooks/`: shared logic
* `lib/`: utilities

### Backend

* `main.py`: FastAPI initialization
* `routers/`: endpoints + agent tools
* `tests/`: unit/integration tests

### Data Pipeline

* `scrapers/`: HuggingFace data retrieval
* `graph/`: graph assembly
* `storage/`: DVC storage
* `config/`: environment settings

---

## Appendix

### Technology Choices Rationale

* **React** for UI flexibility
* **FastAPI** for fast backend development
* **Neo4j** for modeling lineage relationships
* **OpenAI Agents SDK** for structured tool use
* **Docker** for reproducible environments
