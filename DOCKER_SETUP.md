# Docker Setup for DataDetox

This guide explains how to run the entire DataDetox application using Docker Compose.

## Prerequisites

- Docker Desktop installed ([Get Docker](https://www.docker.com/products/docker-desktop))
- Docker Compose (included with Docker Desktop)

## Quick Start

1. **Copy the environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your credentials**:
   ```bash
   # Required: Your OpenAI API key
   OPENAI_API_KEY=sk-proj-...
   
   # Required: Your HuggingFace token
   HF_TOKEN=hf_...
   
   # Required: Your Neo4j credentials
   NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-password
   ```

3. **Start all services**:
   ```bash
   docker-compose up --build
   ```

   Or run in detached mode:
   ```bash
   docker-compose up -d --build
   ```

4. **Access the application**:
   - **Frontend**: http://localhost:3000
   - **Chatbot**: http://localhost:3000/chatbot
   - **Backend API**: http://localhost:8000
   - **API Docs**: http://localhost:8000/docs

## Services

The `docker-compose.yml` starts the following services:

### 1. Backend (Port 8000)
- FastAPI application
- Handles search agent queries
- Connects to Neo4j and HuggingFace

### 2. Frontend (Port 3000)
- React + Vite application
- User interface for the chatbot
- Pre-configured to connect to backend at `http://localhost:8000`

### 3. Neo4j (Ports 7474, 7687) - Optional
- Graph database for model lineage
- Browser UI at http://localhost:7474
- Can use cloud Neo4j instead (configure in `.env`)

### 4. Model Lineage Scraper - Optional
- Scrapes HuggingFace model relationships
- Populates Neo4j database

## Common Commands

### Start services
```bash
docker-compose up
```

### Start in background
```bash
docker-compose up -d
```

### Rebuild containers
```bash
docker-compose up --build
```

### Stop services
```bash
docker-compose down
```

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Restart a service
```bash
docker-compose restart backend
```

## Troubleshooting

### Port already in use
If you get "port already in use" errors:
```bash
# Stop any running processes
lsof -ti:3000 | xargs kill -9
lsof -ti:8000 | xargs kill -9
```

### Environment variables not loading
Make sure:
1. `.env` file exists in the project root
2. All required variables are set
3. No extra quotes around values

### Backend can't connect to Neo4j
Check your Neo4j credentials in `.env`:
- Ensure `NEO4J_URI` includes the protocol (neo4j+s://)
- Verify username and password are correct
- Test connection at https://neo4j.com/cloud

### Frontend shows API errors
1. Check backend is running: `docker-compose ps`
2. View backend logs: `docker-compose logs backend`
3. Test API directly: http://localhost:8000/docs

## Development Mode

For development with hot-reload, you can run services separately:

### Backend only
```bash
docker-compose up backend
```

### Frontend only (assumes backend is running)
```bash
docker-compose up frontend
```

## Production Deployment

For production:
1. Remove volume mounts from `docker-compose.yml`
2. Use production-grade secrets management
3. Set proper Neo4j authentication
4. Configure CORS properly in backend
5. Use environment-specific `.env` files

## File Structure

```
.
├── docker-compose.yml      # Orchestrates all services
├── .env                    # Your secrets (git-ignored)
├── .env.example           # Template for .env
├── backend/
│   ├── Dockerfile         # Backend container config
│   └── .dockerignore      # Files to exclude from build
└── frontend/
    ├── Dockerfile         # Frontend container config
    └── .dockerignore      # Files to exclude from build
```

## Next Steps

After starting the services:
1. Visit http://localhost:3000/chatbot
2. Try example queries like "Tell me about BERT models"
3. Check the backend logs to see the agent in action
4. Explore the API at http://localhost:8000/docs
