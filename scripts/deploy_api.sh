#!/bin/bash
# deploy_api.sh - Deploy the Depression XAI API
# ================================================

set -e

echo "=== Depression XAI API Deployment ==="

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "Docker found. Building and deploying..."
    docker-compose build
    docker-compose up -d
    echo "API deployed at http://localhost:8000"
    echo "MLflow UI at http://localhost:5000"
    echo ""
    echo "Check health: curl http://localhost:8000/health"
else
    echo "Docker not found. Starting with uvicorn..."
    echo "Installing dependencies..."
    pip install -r requirements.txt

    echo "Starting API server..."
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
fi
