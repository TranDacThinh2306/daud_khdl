"""
visualizer.py - Visualization của explanations
=================================================
Visualization tools for SHAP, LIME, and feature importance.
"""

import logging
import os
from typing import Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import shap

logger = logging.getLogger(__name__)


class ExplanationVisualizer:
    """Visualize model explanations and feature importance."""

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_shap_summary(
        self,
        shap_values,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        filename: str = "shap_summary_plot.png",
    ):
        """Generate SHAP summary plot."""
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            show=False,
        )
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved to {path}")

    def plot_feature_importance(
        self,
        importances: dict,
        top_k: int = 20,
        filename: str = "feature_importance_global.png",
    ):
        """Plot global feature importance bar chart."""
        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_k]
        features, values = zip(*sorted_items)

        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        plt.barh(range(len(features)), values, color=colors)
        plt.yticks(range(len(features)), features)
        plt.xlabel("Importance")
        plt.title("Global Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Feature importance plot saved to {path}")

    def plot_lime_explanation(
        self,
        explanation,
        filename: str = "lime_explanation.png",
    ):
        """Save LIME explanation figure."""
        fig = explanation.as_pyplot_figure()
        path = os.path.join(self.output_dir, "lime_examples", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"LIME explanation saved to {path}")

    def plot_shap_waterfall(
        self,
        shap_values,
        index: int = 0,
        filename: str = "shap_waterfall.png",
    ):
        """Plot SHAP waterfall for a single prediction."""
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_values[index], show=False)
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP waterfall plot saved to {path}")
