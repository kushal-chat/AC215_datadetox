#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

export BASE_DIR=$(pwd)
export IMAGE_NAME="datadetox-backend"

# Build the image based on the Dockerfile
echo "Building docker image..."
docker build -t $IMAGE_NAME -f Dockerfile .

# Run Container
echo "Run Docker container..."
docker run --rm --name $IMAGE_NAME -ti \
-v $BASE_DIR:/app \
$IMAGE_NAME