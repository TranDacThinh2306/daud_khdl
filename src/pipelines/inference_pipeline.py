"""
inference_pipeline.py - Real-time prediction + explanation
============================================================
Pipeline for real-time depression prediction with explanations.
"""

import logging
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

from src.data.preprocess import TextPreprocessor
from src.features.text_features import TextFeatureExtractor
from src.features.linguistic_features import LinguisticFeatureExtractor

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Real-time inference pipeline with optional XAI explanations."""

    def __init__(
        self,
        model_path: str,
        text_extractor: Optional[TextFeatureExtractor] = None,
        feature_names: Optional[List[str]] = None,
    ):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.preprocessor = TextPreprocessor()
        self.text_extractor = text_extractor
        self.ling_extractor = LinguisticFeatureExtractor()
        self.feature_names = feature_names
        self.shap_explainer = None

    def load_explainer(self, explainer_path: str):
        """Load a saved SHAP explainer."""
        with open(explainer_path, "rb") as f:
            self.shap_explainer = pickle.load(f)

    def predict(self, text: str, explain: bool = False) -> Dict:
        """
        Predict depression indicators for a single comment.

        Args:
            text: Raw text comment
            explain: Whether to include explanation

        Returns:
            Prediction result with optional explanation
        """
        cleaned = self.preprocessor.clean_text(text)
        ling_features = self.ling_extractor.extract_single(cleaned)
        ling_array = np.array(list(ling_features.values())).reshape(1, -1)

        if self.text_extractor:
            import pandas as pd
            text_features = self.text_extractor.transform(pd.Series([cleaned]))
            X = np.hstack([text_features, ling_array])
        else:
            X = ling_array

        prediction = int(self.model.predict(X)[0])
        result = {
            "text": text,
            "cleaned_text": cleaned,
            "prediction": prediction,
            "label": "depressed" if prediction == 1 else "non-depressed",
        }

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            result["confidence"] = float(max(proba))
            result["probabilities"] = {
                "non-depressed": float(proba[0]),
                "depressed": float(proba[1]),
            }

        if explain and self.shap_explainer:
            result["explanation"] = self.shap_explainer.explain_single(
                X[0], self.feature_names
            )

        return result

    def predict_batch(self, texts: List[str], explain: bool = False) -> List[Dict]:
        """Predict for a batch of texts."""
        return [self.predict(text, explain) for text in texts]
