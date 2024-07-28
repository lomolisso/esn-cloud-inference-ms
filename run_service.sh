#!/bin/bash

# Load the environment variables
source .env

# Run the FastAPI application
uvicorn app.main:app --host 0.0.0.0 --port $INFERENCE_MICROSERVICE_PORT