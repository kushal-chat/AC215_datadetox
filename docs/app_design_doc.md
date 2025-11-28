# DataDetox Application Design Document

**Team Members**: Kushal Chattopadhyay, Keyu Wang, Terry Zhou
**Group Name**: DataDetox

---

## Executive Summary

DataDetox is an AI agent system that traces ML model provenance and training data lineage using HuggingFace metadata and Neo4j graph database. It helps practitioners identify hidden risks (copyrighted data, problematic datasets like LAION-5B) in model dependency chains through natural language queries and interactive graph visualization.

**Technology Stack**: React + TypeScript, FastAPI + Python 3.13, OpenAI Agents SDK, Neo4j, Docker

---

## Solution Architecture

### System Overview

DataDetox consists of three main layers:

**1. User Interface Layer**
- React frontend (Port 3000) with chat interface and graph visualization
- Communicates with backend via REST API

**2. Application Layer**
- FastAPI backend (Port 8000) orchestrating AI agent
- OpenAI Agent with two tools:
  - HuggingFace API tool (model/dataset metadata)
  - Neo4j query tool (lineage relationships)

**3. Data Pipeline Layer**
- HuggingFace scraper → Graph builder → Neo4j loader
- DVC for data versioning

### Data Flow

#### High-Level Flow
```
User Query → Frontend → Backend API → AI Agent
                                       ↓
                        [HuggingFace API + Neo4j Query]
                                       ↓
                        Agent synthesizes response
                                       ↓
                Frontend ← { result, graph_data }
                                       ↓
                    Display chat + visualization
```

#### Detailed Query Flow

```
+---------------+
|  User Input   |
|  "Tell me     |
|  about BERT"  |
+-------+-------+
        |
        v
+---------------------+
|  Frontend Chatbot   |
|  - Validate input   |
|  - Show "thinking"  |
+----------+----------+
           | POST /flow/search
           | { query_val: "..." }
           v
+---------------------+
|  Backend API        |
|  - Initialize state |
|  - Store context    |
+----------+----------+
           |
           v
+---------------------+
|  Search Agent       |
|  - Parse query      |
|  - Plan tool use    |
+----------+----------+
           |
           |
    +------+------+
    |             |
    v             v
+--------------+  +--------------+
| HF Tool      |  | Neo4j Tool   |
| - Get model  |  | - Get graph  |
|   metadata   |  |   data       |
+------+-------+  +------+-------+
       |                 |
       +--------+--------+
                |
                v
       +------------------+
       | Agent Synthesis  |
       | - Combine results|
       | - Generate reply |
       +--------+---------+
                |
                v
       +------------------+
       | Backend Response |
       | {                |
       |   result: str,   |
       |   neo4j_data: {} |
       | }                |
       +--------+---------+
                |
                v
       +------------------+
       | Frontend Update  |
       | - Display message|
       | - Render graph   |
       +------------------+
```

### Component Interactions

**Step-by-step execution:**

1. **User submits query** (e.g., "Tell me about qwen3-4b")
   - Frontend validates input and displays "thinking" indicator

2. **Frontend sends POST request** to `/flow/search`
   - Request body: `{ "query_val": "Tell me about qwen3-4b" }`

3. **Backend initializes request state**
   - Creates tool results storage
   - Stores original query for context

4. **Agent searches HuggingFace** (Tool 1)
   - Queries HuggingFace API for model "qwen3-4b"
   - Extracts model_id and metadata
   - Returns: `{ model_id: "Qwen/Qwen3-4B", downloads: 1.5M, ... }`

5. **Agent queries Neo4j** (Tool 2)
   - Executes Cypher query with model_id
   - Retrieves connected models and datasets
   - Stores graph data in request state
   - Returns: `{ nodes: [...], relationships: [...] }`

6. **Agent fetches details** (Tool 1 again)
   - Searches HuggingFace for connected model/dataset IDs
   - Enriches graph nodes with full metadata

7. **Agent generates summary**
   - Synthesizes findings into natural language
   - Identifies any problematic datasets
   - Provides risk assessment

8. **Backend returns response**
   - `{ "result": "summary text", "neo4j_data": {...} }`

9. **Frontend updates UI**
   - Removes "thinking" indicator
   - Displays agent's text response in chat
   - Renders interactive graph visualization
   - Shows toast notification with metrics

### System-Wide Workflow

```
[Data Collection Phase]
+----------------------+
| HuggingFace Scraper  |
| - Scrape models/data |
+----------+-----------+
           |
           v
+----------------------+
| DVC + Neo4j          |
| - Version data       |
| - Build graph        |
+----------+-----------+
           |
           v
    [Graph Ready]

[Query Processing Phase]
+----------------------+
| User Query           |
+----------+-----------+
           |
           v
+----------------------+
| React Frontend       |
| - Validate           |
| - POST request       |
+----------+-----------+
           |
           v
+----------------------+
| FastAPI Backend      |
| - Agent orchestrate  |
+----------+-----------+
           |
    +------+------+
    |             |
    v             v
+--------+    +--------+
| HF API |    | Neo4j  |
+--------+    +--------+
    |             |
    +------+------+
           |
           v
+----------------------+
| Response + Graph     |
+----------+-----------+
           |
           v
+----------------------+
| Frontend Display     |
| - Chat + Tree viz    |
+----------------------+
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | React 18 + Vite + TypeScript | UI and visualization |
| Backend API | FastAPI + Python 3.13 | REST API and orchestration |
| AI Agent | OpenAI Agents SDK | Query processing, tool coordination |
| Graph Database | Neo4j 5.15 | Model lineage storage |
| Data Pipeline | Python + HuggingFace Hub API | Data collection |
| Versioning | DVC | Data reproducibility |
| Deployment | Docker Compose | Service orchestration |

---

## Technical Architecture

### Frontend Architecture

**Stack**: React 18.3, Vite 7.2, TypeScript 5.8, TailwindCSS 3.4, React-D3-Tree 3.6

**Structure**:
```
src/
├── components/
│   ├── ui/              // Radix UI components
│   ├── ChatMessage.tsx  // Message display
│   └── ModelTree.tsx    // Graph visualization
├── pages/
│   ├── Index.tsx        // Landing page
│   └── Chatbot.tsx      // Main interface
└── hooks/
    └── use-mobile.tsx   // Responsive utilities
```

**State Management**: React Hooks + TanStack Query for API calls

**Key Features**:
- Resizable two-panel layout (chat | graph)
- Real-time message updates with thinking indicators
- Interactive D3 tree visualization with zoom/pan
- Markdown rendering for agent responses

### Backend Architecture

**Stack**: FastAPI 0.121, Python 3.13, OpenAI Agents SDK 0.3, Neo4j driver 6.0

**Structure**:
```
backend/
├── main.py              # FastAPI app entry
├── routers/
│   └── search/
│       ├── agent.py     # Agent configuration
│       └── utils/
│           ├── huggingface.py    # HF API tool
│           ├── search_neo4j.py   # Neo4j tool
│           └── tool_state.py     # Context management
└── tests/
```

**Agent Configuration**:
```python
instructions = """
1. search_huggingface() to get model/dataset info and ID
2. search_neo4j(model_id) to get connected models/datasets
3. search_huggingface() for details on connections
4. Summarize findings with risk assessment
"""

tools = [search_huggingface, search_neo4j]
agent = Agent(name="SearchAgent", instructions=instructions,
              model="gpt-5-nano", tools=tools)
```

**API Endpoint**:
- `POST /flow/search` - Execute agent query
  - Input: `{ "query_val": "string" }`
  - Output: `{ "result": "string", "neo4j_data": {...} }`

### Data Pipeline Architecture

**Stack**: HuggingFace Hub 0.20, Neo4j 5.15, DVC 3.40, Pydantic 2.5

**Pipeline Stages**:

1. **Data Collection**
   - Scrape HuggingFace models and datasets
   - Extract relationships (TRAINED_ON, DERIVED_FROM)
   - Save to JSON with DVC versioning

2. **Graph Construction**
   - Load scraped data
   - Build nodes (models, datasets) and edges (relationships)
   - Validate graph structure

3. **Neo4j Loading**
   - Batch import nodes and relationships
   - Create indexes for performance

**Data Scraping Flow**:

```
+--------------------+
|  Manual Trigger    |
|  or Scheduled      |
+---------+----------+
          |
          v
+-------------------------------+
|  HuggingFace Scraper          |
|  - Fetch model list           |
|  - Iterate through models     |
|  - Extract relationships      |
+--------------+----------------+
               |
               v
+-------------------------------+
|  Data Validation              |
|  - Check required fields      |
|  - Deduplicate entries        |
+--------------+----------------+
               |
               v
+-------------------------------+
|  DVC Storage                  |
|  - Save JSON files            |
|  - Create DVC tracking        |
|  - Commit metadata            |
+--------------+----------------+
               |
               v
+-------------------------------+
|  Graph Builder                |
|  - Load from DVC              |
|  - Build node/edge lists      |
|  - Validate graph structure   |
+--------------+----------------+
               |
               v
+-------------------------------+
|  Neo4j Client                 |
|  - Clear old data (optional)  |
|  - Batch import nodes         |
|  - Batch import edges         |
|  - Create indexes             |
+-------------------------------+
```

**Graph Schema**:
```cypher
// Nodes
(:Model {model_id, downloads, likes, pipeline_tag, library_name, created_at, tags, url})
(:Dataset {dataset_id, downloads, likes, tags, url, is_problematic})

// Relationships
(:Model)-[:TRAINED_ON]->(:Dataset)
(:Model)-[:DERIVED_FROM]->(:Model)
```

### Design Patterns

**Backend**:
- **Dependency Injection**: FastAPI request context for tool state
- **Tool Pattern**: Modular functions as agent tools
- **Repository Pattern**: Neo4j client abstracts database operations

**Frontend**:
- **Component Composition**: Radix UI primitives + custom components
- **Container/Presentational**: Pages orchestrate, components display
- **Custom Hooks**: Shared logic (mobile detection, API calls)

**Data Pipeline**:
- **ETL Pattern**: Extract (HF) → Transform (graph builder) → Load (Neo4j)
- **Versioning Strategy**: DVC tracks data snapshots with Git integration

### Deployment Architecture

**Docker Compose Services**:

```yaml
services:
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    environment:
      - VITE_API_URL=http://localhost:8000

  backend:
    build: ./backend
    ports: ["8000:8000"]
    environment:
      - OPENAI_API_KEY
      - HF_TOKEN
      - NEO4J_URI
      - NEO4J_USER
      - NEO4J_PASSWORD

  # Optional local Neo4j (can use cloud Neo4j Aura)
  neo4j:
    image: neo4j:5.15-community
    ports: ["7474:7474", "7687:7687"]
```

**Planned Production Deployment**:
- **Platform**: Google Cloud Platform or AWS
- **Frontend**: Cloud Run / ECS (containerized)
- **Backend**: Cloud Run / ECS (containerized)
- **Database**: Neo4j Aura (managed)
- **Storage**: Cloud Storage / S3 (DVC remote)
- **Secrets**: Secret Manager


### Security Considerations

- **CORS**: Configured for localhost:3000 in development
- **Input Validation**: Pydantic models for API requests
- **Injection Prevention**: Parameterized Cypher queries
- **Secrets Management**: Environment variables, gitignored .env files
- **Rate Limiting**: Planned for production deployment

### Testing Strategy

**Backend**: pytest with 80%+ coverage target
- Unit tests for HuggingFace and Neo4j tools
- Integration tests for API endpoints
- Mock external services (HF API, Neo4j)

**Frontend**: Component testing for critical paths
- Chat message rendering
- Graph visualization updates
- API error handling

**CI/CD Pipeline Flow**:

```
+------------------+
| Code Push/PR     |
+--------+---------+
         |
         v
+------------------+
| GitHub Actions   |
| Trigger          |
+--------+---------+
         |
         +-------------------+
         |                   |
         v                   v
+----------------+  +------------------+
| Backend Tests  |  | Frontend Lint    |
| - pytest       |  | - eslint         |
| - coverage     |  |                  |
+-------+--------+  +--------+---------+
        |                    |
        v                    v
+----------------+  +------------------+
| Model-Lineage  |  | Build Check      |
| Tests          |  | - Docker build   |
+-------+--------+  +--------+---------+
        |                    |
        +--------+-----------+
                 |
                 v
        +------------------+
        | All Tests Pass?  |
        +--------+---------+
                 |
         +-------+-------+
         |               |
         v               v
      [YES]           [FAIL]
         |               |
         v               v
+----------------+  +------------------+
| Merge to Main  |  | Notify Developer |
+-------+--------+  | - Failed tests   |
        |           | - Coverage drop  |
        v           +------------------+
+----------------+
| Deploy (Prod)  |
| - Build images |
| - Push registry|
| - Update K8s   |
+----------------+
```

**CI/CD Stages:**
- **Linting**: ruff (Python), eslint (TypeScript)
- **Testing**: pytest for backend and model-lineage with coverage reports
- **Build Validation**: Docker image builds successfully
- **Deployment**: Automated deployment to production on main branch merge

---

## User Interface Design

### Chatbot Interface

**Layout**: Resizable two-panel design
- **Left Panel** (30-70% width): Chat conversation
  - Message history with timestamps
  - User input field with send button
  - Thinking indicators during processing

- **Right Panel** (30-70% width): Model dependency tree
  - Interactive D3 hierarchical visualization
  - Expandable/collapsible nodes
  - Zoom and pan controls
  - Node tooltips showing metadata

![Chatbot UI](../assets/img/ms4/chatbot_ui.png)

*Example. Qwen3-4B Lineage Graph (Right Panel)*

* **🟡 Yellow node:** The queried model (**Qwen/Qwen3-4B**).
* **🔵 Blue nodes:** Related models, including the upstream base model and downstream finetuned variants.
* **⚪ White nodes:** Datasets used to train or finetune specific models.

A compact overview of model lineage, dependencies, and dataset provenance.



## Code Organization

### Frontend Structure
- **components/**: Reusable UI components (Radix UI + custom)
- **pages/**: Route components (Index, Chatbot, NotFound)
- **hooks/**: Shared React hooks (mobile detection)
- **lib/**: Utilities (TailwindCSS helpers)

### Backend Structure
- **main.py**: FastAPI app initialization, CORS, router registration
- **routers/**: API route handlers
  - **search/**: Agent-related endpoints and tools
- **tests/**: Unit and integration tests

### Data Pipeline Structure
- **scrapers/**: HuggingFace API interaction
- **graph/**: Graph construction and Neo4j client
- **storage/**: DVC data store management
- **config/**: Environment settings (Pydantic)

---

## Appendix

### Technology Rationale

- **React**: Large ecosystem, TypeScript support, rich visualization libraries
- **FastAPI**: High performance (async), automatic OpenAPI docs, Pydantic validation
- **Neo4j**: Native graph database, efficient relationship queries, Cypher language
- **OpenAI Agents SDK**: Built-in tool orchestration, context management, streaming
- **Docker**: Reproducible environments, easy local development

### References

1. HuggingFace Hub API: https://huggingface.co/docs/hub
2. Neo4j Documentation: https://neo4j.com/docs/
3. OpenAI Agents SDK: https://github.com/openai/swarm
4. FastAPI Documentation: https://fastapi.tiangolo.com/
5. React Documentation: https://react.dev/

### Team Contributions

| Team Member | Responsibilities |
|-------------|-----------------|
| Kushal Chattopadhyay | Backend, Agent orchestration |
| Keyu Wang | Frontend, UI/UX design |
| Terry Zhou | Data pipeline, Neo4j integration |

---

**Document Version**: 1.0
**Date**: 2025-01-28
**Authors**: DataDetox Team

---

**End of Document**
