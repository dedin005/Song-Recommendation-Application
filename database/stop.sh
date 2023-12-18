#!/bin/bash

# Find the container ID for Qdrant
CONTAINER_ID=$(docker ps -q --filter ancestor=qdrant/qdrant)

# Stop the container
if [ -n "$CONTAINER_ID" ]; then
    docker stop $CONTAINER_ID
else
    echo "No running Qdrant container found."
fi

