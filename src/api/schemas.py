"""
schemas.py - Pydantic models cho request/response
====================================================
Request and response schemas for the API.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for single prediction."""
    text: str = Field(..., description="Social media comment text", min_length=1)


class PredictResponse(BaseModel):
    """Response schema for prediction."""
    prediction: int = Field(..., description="0=non-depressed, 1=depressed")
    label: str = Field(..., description="Human-readable label")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")


class ExplainRequest(BaseModel):
    """Request schema for prediction with explanation."""
    text: str = Field(..., description="Social media comment text", min_length=1)
    method: str = Field("shap", description="Explanation method: 'shap' or 'lime'")
    num_features: int = Field(10, description="Number of top features to explain")


class ExplainResponse(BaseModel):
    """Response schema for prediction with explanation."""
    prediction: int
    label: str
    confidence: Optional[float] = None
    explanation: Dict = Field(default_factory=dict, description="Feature contributions")


class BatchRequest(BaseModel):
    """Request schema for batch prediction."""
    texts: List[str] = Field(..., description="List of comment texts", min_length=1)
    explain: bool = Field(False, description="Include explanations")


class BatchResponse(BaseModel):
    """Response schema for batch prediction."""
    predictions: List[PredictResponse]
    total: int
