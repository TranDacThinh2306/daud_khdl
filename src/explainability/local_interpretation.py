"""
local_interpretation.py - Local explanation cho 1 comment
============================================================
Generate local explanation for individual comment predictions.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class LocalInterpreter:
    """Generate interpretable explanations for individual predictions."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def interpret(
        self,
        instance: np.ndarray,
        shap_explainer=None,
        lime_explainer=None,
        top_k: int = 10,
    ) -> Dict:
        """
        Generate a local interpretation for a single instance.

        Returns:
            Dictionary with prediction, confidence, and feature contributions
        """
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        prediction = self.model.predict(instance)[0]
        confidence = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(instance)[0]
            confidence = float(max(proba))

        result = {
            "prediction": int(prediction),
            "label": "depressed" if prediction == 1 else "non-depressed",
            "confidence": confidence,
            "top_features": {},
        }

        # SHAP explanation
        if shap_explainer:
            shap_contrib = shap_explainer.explain_single(instance[0], self.feature_names)
            sorted_features = sorted(shap_contrib.items(), key=lambda x: abs(x[1]), reverse=True)
            result["shap_contributions"] = dict(sorted_features[:top_k])

        # LIME explanation
        if lime_explainer:
            predict_fn = getattr(self.model, "predict_proba", self.model.predict)
            exp = lime_explainer.explain_instance(instance[0], predict_fn)
            result["lime_contributions"] = lime_explainer.get_top_features(exp)

        return result

    def format_explanation(self, interpretation: Dict) -> str:
        """Format interpretation as human-readable text."""
        lines = [
            f"Prediction: {interpretation['label']}",
            f"Confidence: {interpretation['confidence']:.2%}" if interpretation["confidence"] else "",
            "\nTop contributing features:",
        ]
        contributions = interpretation.get("shap_contributions", interpretation.get("lime_contributions", {}))
        for feature, value in list(contributions.items())[:10]:
            direction = "↑" if value > 0 else "↓"
            lines.append(f"  {direction} {feature}: {value:.4f}")
        return "\n".join(lines)
