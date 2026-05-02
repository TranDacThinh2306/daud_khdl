"""
stability.py - Kiểm tra stability của explanations
=====================================================
Check consistency and stability of SHAP/LIME explanations.
"""

import logging
from typing import Any, Callable, List, Optional

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class StabilityChecker:
    """Check stability and consistency of XAI explanations."""

    def __init__(self, n_repeats: int = 5, random_state: int = 42):
        self.n_repeats = n_repeats
        self.random_state = random_state

    def check_lime_stability(
        self,
        instance: np.ndarray,
        lime_explainer,
        predict_fn: Callable,
        num_features: int = 10,
    ) -> dict:
        """
        Check LIME explanation stability by running multiple times.

        Returns:
            Dictionary with stability metrics
        """
        explanations = []
        for i in range(self.n_repeats):
            exp = lime_explainer.explain_instance(
                instance, predict_fn, num_features=num_features
            )
            feature_weights = dict(exp.as_list())
            explanations.append(feature_weights)

        # Check feature overlap across runs
        all_features = [set(exp.keys()) for exp in explanations]
        intersection = set.intersection(*all_features) if all_features else set()
        union = set.union(*all_features) if all_features else set()
        jaccard = len(intersection) / len(union) if union else 1.0

        # Check rank correlation
        correlations = []
        for i in range(len(explanations) - 1):
            common = set(explanations[i].keys()) & set(explanations[i + 1].keys())
            if len(common) >= 3:
                v1 = [explanations[i][f] for f in common]
                v2 = [explanations[i + 1][f] for f in common]
                corr, _ = spearmanr(v1, v2)
                correlations.append(corr)

        return {
            "jaccard_similarity": jaccard,
            "avg_rank_correlation": np.mean(correlations) if correlations else None,
            "n_consistent_features": len(intersection),
            "n_total_features": len(union),
            "is_stable": jaccard > 0.7,
        }

    def check_shap_stability(
        self,
        instances: np.ndarray,
        shap_explainer,
        noise_level: float = 0.01,
    ) -> dict:
        """
        Check SHAP stability by adding small noise perturbations.

        Returns:
            Dictionary with stability metrics
        """
        original = shap_explainer.explain_single(instances[0])
        original_vals = np.array(list(original.values()))

        correlations = []
        for _ in range(self.n_repeats):
            noise = np.random.normal(0, noise_level, instances[0].shape)
            perturbed = instances[0] + noise
            perturbed_exp = shap_explainer.explain_single(perturbed)
            perturbed_vals = np.array(list(perturbed_exp.values()))

            if len(original_vals) >= 3:
                corr, _ = spearmanr(original_vals, perturbed_vals)
                correlations.append(corr)

        return {
            "avg_perturbation_correlation": np.mean(correlations) if correlations else None,
            "min_correlation": np.min(correlations) if correlations else None,
            "is_stable": np.mean(correlations) > 0.8 if correlations else False,
        }
