"""
global_interpretation.py - Global feature importance
======================================================
Aggregate explanations for global model interpretation.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GlobalInterpreter:
    """Generate global interpretations from aggregate explanations."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.global_importance = None

    def compute_from_shap(self, shap_values: np.ndarray) -> pd.DataFrame:
        """Compute global feature importance from SHAP values."""
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        importance = np.abs(shap_values).mean(axis=0)
        self.global_importance = pd.DataFrame({
            "feature": self.feature_names[:len(importance)],
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        logger.info("Global feature importance computed from SHAP")
        return self.global_importance

    def compute_from_model(self, model: Any) -> pd.DataFrame:
        """Extract feature importance directly from tree-based models."""
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_).flatten()
        else:
            raise ValueError("Model does not expose feature importances")

        self.global_importance = pd.DataFrame({
            "feature": self.feature_names[:len(importance)],
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return self.global_importance

    def get_top_features(self, k: int = 20) -> pd.DataFrame:
        """Get top-k most important features."""
        if self.global_importance is None:
            raise RuntimeError("Compute importance first")
        return self.global_importance.head(k)

    def get_category_importance(self, category_map: Dict[str, List[str]]) -> pd.DataFrame:
        """Aggregate importance by feature category."""
        if self.global_importance is None:
            raise RuntimeError("Compute importance first")

        results = []
        for category, features in category_map.items():
            mask = self.global_importance["feature"].isin(features)
            cat_importance = self.global_importance.loc[mask, "importance"].sum()
            results.append({"category": category, "importance": cat_importance})
        return pd.DataFrame(results).sort_values("importance", ascending=False)
