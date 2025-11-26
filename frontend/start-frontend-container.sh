#!/bin/bash

# Exit on error and undefined variables
set -eu

# Constants
IMAGE_NAME="datadetox-frontend"
CONTAINER_NAME="datadetox-frontend"
PORT="8080"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper function for logging
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    error "Docker is not running. Please start Docker first."
fi

# Clean up any existing container (ignore errors with || true)
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true
log "Cleaned up any existing containers"

# Build the image
log "Building Docker image..."
if ! docker build -t $IMAGE_NAME -f Dockerfile .; then
    error "Failed to build Docker image"
fi

# Run the container
log "Starting container..."
docker run \
    --rm \
    --name $CONTAINER_NAME \
    -p $PORT:$PORT \
    -d \
    $IMAGE_NAME

log "Container started successfully!"
log "Frontend available at http://localhost:$PORT"

# Watch container logs
log "Showing container logs (Ctrl+C to exit)..."
docker logs -f $CONTAINER_NAME
