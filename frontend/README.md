# DataDetox Frontend

The frontend application for DataDetox, an interactive AI agent orchestration system for tracing ML data and model provenance. This React-based web application provides a user-friendly interface for exploring AI model lineages, identifying hidden risks in datasets, and understanding upstream dependencies in the AI ecosystem.

## Overview

The DataDetox frontend enables users to:
- Query AI models and datasets through an interactive chatbot interface
- Visualize model lineage trees and relationships
- Explore connections between models, datasets, and upstream dependencies
- Identify potential risks in model training data (e.g., copyrighted content, problematic datasets)

## Technology Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and development server
- **Tailwind CSS** - Styling
- **shadcn-ui** - UI component library (Radix UI primitives)
- **React Router** - Client-side routing
- **TanStack Query (React Query)** - Data fetching and state management
- **React Flow / ELK.js** - Graph visualization for model trees
- **React Markdown** - Markdown rendering for chat messages

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── ui/             # shadcn-ui components
│   │   ├── ChatMessage.tsx # Chat message display
│   │   ├── ModelTree*.tsx  # Model lineage visualization
│   │   ├── Navbar.tsx      # Navigation bar
│   │   ├── Hero.tsx        # Landing page hero section
│   │   ├── Features.tsx    # Features showcase
│   │   └── UseCases.tsx    # Use cases section
│   ├── pages/              # Page components
│   │   ├── Index.tsx       # Landing page
│   │   ├── Chatbot.tsx     # Main chatbot interface
│   │   └── NotFound.tsx    # 404 page
│   ├── hooks/              # Custom React hooks
│   ├── lib/                # Utility functions
│   └── App.tsx             # Main app component with routing
├── public/                 # Static assets
└── package.json            # Dependencies and scripts
```

## Getting Started

### Prerequisites

- **Node.js** (v18 or higher) - [Install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)
- **npm** or **bun** (comes with Node.js)

### Installation

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

4. **Access the application**:
   - Frontend: http://localhost:3000
   - Chatbot: http://localhost:3000/chatbot

### Development Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run build:dev` - Build in development mode
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint
- `npm run test` - Run tests with Vitest
- `npm run test:ui` - Run tests with UI
- `npm run test:coverage` - Run tests with coverage report

## Features

### Landing Page (`/`)
- Hero section introducing DataDetox
- Features showcase
- Use cases demonstration
- Navigation to chatbot interface

### Chatbot Interface (`/chatbot`)
- Interactive chat interface for querying models and datasets
- Real-time model lineage tree visualization
- Support for complex queries about model relationships
- Display of metadata (arXiv papers, search terms, processing times)
- Risk indicators for datasets

### Model Visualization
- Interactive graph visualization of model relationships
- Tree layout using ELK.js
- Node details with model metadata
- Relationship types between models and datasets

## Backend Integration

The frontend communicates with the backend API through a Vite proxy configuration:

- **Development**: Requests to `/backend/*` are proxied to `http://localhost:8000`
- **Production**: Configure the backend URL via environment variables

The backend API provides:
- Search endpoints for querying models
- Neo4j graph data for lineage visualization
- HuggingFace model information
- arXiv paper extraction

## Environment Configuration

For production deployments, configure the backend API URL:

```bash
VITE_API_URL=http://your-backend-url:8000
```

## Testing

The project uses Vitest for unit testing and React Testing Library for component testing:

```bash
# Run all tests
npm run test

# Run tests with UI
npm run test:ui

# Generate coverage report
npm run test:coverage
```

## Building for Production

1. **Build the application**:
   ```bash
   npm run build
   ```

2. **Output**: The production build will be in the `dist/` directory

3. **Preview the build**:
   ```bash
   npm run preview
   ```

## Docker Deployment

The frontend can be deployed using Docker. See the root `docker-compose.yml` and `Dockerfile` in this directory for containerization configuration.

## Contributing

When contributing to the frontend:

1. Follow the existing code style and TypeScript conventions
2. Add tests for new components and features
3. Ensure all tests pass before submitting PRs
4. Update this README if adding new features or changing setup

## Related Documentation

- [Main Project README](../README.md) - Full project overview and setup
- [Backend README](../backend/README.md) - Backend API documentation
- [Application Design Document](../docs/ms4/app_design_doc.md) - System architecture
