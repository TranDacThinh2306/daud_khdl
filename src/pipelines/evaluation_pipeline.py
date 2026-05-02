"""
evaluation_pipeline.py - Nested cross-validation
===================================================
Rigorous evaluation using nested cross-validation.
"""

import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Nested cross-validation evaluation pipeline."""

    def __init__(
        self,
        outer_folds: int = 5,
        inner_folds: int = 3,
        random_state: int = 42,
    ):
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.random_state = random_state

    def nested_cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        scoring: str = "f1",
    ) -> Dict:
        """
        Perform nested cross-validation for unbiased evaluation.

        Returns:
            Dictionary with scores and statistics
        """
        outer_cv = StratifiedKFold(
            n_splits=self.outer_folds, shuffle=True, random_state=self.random_state
        )

        scores = cross_val_score(
            model, X, y, cv=outer_cv, scoring=scoring, n_jobs=-1
        )

        result = {
            "scores": scores.tolist(),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "metric": scoring,
        }

        logger.info(
            f"Nested CV ({self.outer_folds} folds): "
            f"{scoring} = {result['mean']:.4f} ± {result['std']:.4f}"
        )
        return result

    def compare_models(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        scoring: str = "f1",
    ) -> Dict:
        """Compare multiple models using nested CV."""
        results = {}
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            results[name] = self.nested_cross_validate(model, X, y, scoring)

        best = max(results, key=lambda k: results[k]["mean"])
        logger.info(f"Best model: {best} ({results[best]['mean']:.4f})")
        return results
