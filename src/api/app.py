"""
app.py - FastAPI server
=========================
Main FastAPI application for depression detection API.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.api.middleware import LoggingMiddleware

logger = logging.getLogger(__name__)

# Global model reference
model_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model_pipeline
    try:
        from src.pipelines.inference_pipeline import InferencePipeline
        model_pipeline = InferencePipeline(
            model_path="models_saved/production/best_model.pkl"
        )
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.warning("No production model found. API running without model.")
    yield
    logger.info("Shutting down API")


app = FastAPI(
    title="Depression XAI API",
    description="Explainable AI API for depression detection in social media comments",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(LoggingMiddleware)

# Routes
app.include_router(router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
    }
