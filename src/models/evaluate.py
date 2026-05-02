"""
evaluate.py - Đánh giá model (accuracy, F1, AUC)
===================================================
Comprehensive model evaluation with multiple metrics.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate classification models with comprehensive metrics."""

    def __init__(self):
        self.results_ = {}

    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate a model on test data."""
        name = model_name or model.__class__.__name__
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
        }

        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            except ValueError:
                metrics["roc_auc"] = None

        self.results_[name] = metrics
        logger.info(f"Evaluation results for {name}:")
        for k, v in metrics.items():
            if v is not None:
                logger.info(f"  {k}: {v:.4f}")

        return metrics

    def get_classification_report(
        self, model: Any, X_test: np.ndarray, y_test: np.ndarray
    ) -> str:
        """Get detailed classification report."""
        y_pred = model.predict(X_test)
        return classification_report(
            y_test, y_pred, target_names=["non-depressed", "depressed"]
        )

    def get_confusion_matrix(
        self, model: Any, X_test: np.ndarray, y_test: np.ndarray
    ) -> np.ndarray:
        """Get confusion matrix."""
        y_pred = model.predict(X_test)
        return confusion_matrix(y_test, y_pred)

    def compare_models(self) -> Dict:
        """Compare all evaluated models."""
        if not self.results_:
            logger.warning("No models evaluated yet")
            return {}
        return self.results_
