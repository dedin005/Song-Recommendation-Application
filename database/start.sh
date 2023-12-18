#!/bin/bash

# Start Qdrant container with custom configuration, telemetry disabled, and TLS enabled
docker run -d -e QDRANT__TELEMETRY_DISABLED=true -p 6333:6333 \
    -v $(pwd)/config.yaml:/qdrant/config/production.yaml \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

exit 0
