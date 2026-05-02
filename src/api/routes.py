"""
routes.py - Endpoints: /predict, /explain, /batch
====================================================
API route definitions for depression detection.
"""

import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    ExplainRequest,
    ExplainResponse,
    BatchRequest,
    BatchResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["prediction"])


def _get_pipeline():
    """Get the loaded model pipeline."""
    from src.api.app import model_pipeline
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_pipeline


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict depression indicators for a single comment."""
    pipeline = _get_pipeline()
    result = pipeline.predict(request.text, explain=False)
    return PredictResponse(
        prediction=result["prediction"],
        label=result["label"],
        confidence=result.get("confidence"),
        probabilities=result.get("probabilities"),
    )


@router.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """Predict with SHAP/LIME explanation."""
    pipeline = _get_pipeline()
    result = pipeline.predict(request.text, explain=True)
    return ExplainResponse(
        prediction=result["prediction"],
        label=result["label"],
        confidence=result.get("confidence"),
        explanation=result.get("explanation", {}),
    )


@router.post("/batch", response_model=BatchResponse)
async def batch_predict(request: BatchRequest):
    """Batch prediction for multiple comments."""
    pipeline = _get_pipeline()
    results = pipeline.predict_batch(request.texts, explain=request.explain)
    predictions = [
        PredictResponse(
            prediction=r["prediction"],
            label=r["label"],
            confidence=r.get("confidence"),
            probabilities=r.get("probabilities"),
        )
        for r in results
    ]
    return BatchResponse(predictions=predictions, total=len(predictions))
