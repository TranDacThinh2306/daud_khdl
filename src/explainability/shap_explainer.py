"""
shap_explainer.py - SHAP wrapper
==================================
SHAP-based model explanation for depression classification.
Supports both tabular (sklearn) and text (transformer) models.
"""

import logging
from typing import Any, Callable, Optional, List

import numpy as np
import shap
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP wrapper for explaining depression predictions (tabular + text)."""

    def __init__(self, model: Any, explainer_type: str = "auto"):
        """
        Args:
            model: Trained model hoặc callable predict function
            explainer_type: 'tree', 'kernel', 'linear', 'text', or 'auto'
        """
        self.model = model
        self.explainer_type = explainer_type
        self.explainer = None
        self.shap_values = None

    # ── Tabular: fit + explain ──

    def fit(self, X_background: np.ndarray, max_samples: int = 100):
        """Initialize the SHAP explainer with background data (tabular)."""
        if X_background.shape[0] > max_samples:
            idx = np.random.choice(X_background.shape[0], max_samples, replace=False)
            X_background = X_background[idx]

        if self.explainer_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.explainer_type == "kernel":
            self.explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
        elif self.explainer_type == "linear":
            self.explainer = shap.LinearExplainer(self.model, X_background)
        elif self.explainer_type == "auto":
            if hasattr(self.model, "estimators_"):
                self.explainer = shap.TreeExplainer(self.model)
            else:
                predict_fn = getattr(self.model, "predict_proba", self.model.predict)
                self.explainer = shap.KernelExplainer(predict_fn, X_background)
        logger.info(f"SHAP explainer initialized: {self.explainer.__class__.__name__}")
        return self

    # ── Text: fit cho transformer models ──

    def fit_text(self, predict_fn: Callable, tokenizer: Any, output_names: Optional[List[str]] = None):
        """
        Initialize SHAP PartitionExplainer for text (transformer models).
        
        Args:
            predict_fn: Callable nhận List[str] → np.ndarray (probabilities)
            tokenizer: Tokenizer dùng để tạo masker
            output_names: Tên các class, ví dụ ['Non-depression', 'Depression']
        """
        masker = shap.maskers.Text(tokenizer)
        self.explainer = shap.Explainer(
            predict_fn, masker,
            output_names=output_names or ['Non-depression', 'Depression']
        )
        self.explainer_type = "text"
        logger.info("SHAP text (Partition) explainer initialized")
        return self

    def explain(self, X, max_evals: int = 500) -> shap.Explanation:
        """
        Generate SHAP explanations.
        
        Args:
            X: np.ndarray (tabular) hoặc List[str] (text)
            max_evals: Max evaluations (chỉ dùng cho text explainer)
        """
        if self.explainer is None:
            raise RuntimeError("Call fit() or fit_text() first")

        if self.explainer_type == "text":
            self.shap_values = self.explainer(X, max_evals=max_evals)
        else:
            self.shap_values = self.explainer(X)
        return self.shap_values

    def get_feature_importance(
        self, X, feature_names: Optional[List[str]] = None,
        target_class: str = 'Depression', max_evals: int = 500
    ) -> dict:
        """
        Get global feature importance from SHAP values.
        Dùng cho cả tabular và text.
        """
        shap_vals = self.explain(X, max_evals=max_evals)

        if self.explainer_type == "text":
            # Text: tính mean |SHAP| cho mỗi token trên toàn dataset
            all_tokens = {}
            for i in range(len(shap_vals)):
                sv = shap_vals[i, :, target_class]
                for token, value in zip(sv.data, sv.values):
                    token = str(token).strip().lower()
                    if token:
                        if token not in all_tokens:
                            all_tokens[token] = []
                        all_tokens[token].append(abs(value))
            # Tính trung bình
            importance = {t: float(np.mean(v)) for t, v in all_tokens.items()}
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        else:
            # Tabular
            if hasattr(shap_vals, "values"):
                vals = np.abs(shap_vals.values).mean(axis=0)
            else:
                vals = np.abs(shap_vals).mean(axis=0)
            if vals.ndim > 1:
                vals = vals[:, 1]
            if feature_names and len(feature_names) == len(vals):
                return dict(zip(feature_names, vals))
            return dict(enumerate(vals))

    def explain_single(self, x, feature_names: Optional[List[str]] = None) -> dict:
        """Explain a single prediction."""
        if isinstance(x, str):
            # Text mode
            sv = self.explain([x])
            return {str(t): float(v) for t, v in zip(sv[0].data, sv[0, :, 'Depression'].values) if str(t).strip()}

        # Tabular mode
        if x.ndim == 1:
            x = x.reshape(1, -1)
        shap_vals = self.explainer.shap_values(x)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        vals = shap_vals[0]

        if feature_names and len(feature_names) == len(vals):
            return dict(zip(feature_names, vals.tolist()))
        return dict(enumerate(vals.tolist()))

    # ── Visualization helpers ──

    def plot_global_summary(self, output_dir: str, top_k: int = 20, filename: str = "shap_global_importance.png"):
        """Plot global bar chart từ đã có shap_values."""
        if self.shap_values is None:
            raise RuntimeError("Call explain() first")

        os.makedirs(output_dir, exist_ok=True)

        if self.explainer_type == "text":
            plt.figure(figsize=(12, 8))
            shap.plots.bar(self.shap_values[:, :, 'Depression'], max_display=top_k, show=False)
            plt.title('SHAP — Global Feature Importance (Depression)')
        else:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(self.shap_values, show=False)
            plt.title('SHAP Summary Plot')

        plt.tight_layout()
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP global summary saved to {path}")
        return path

    def plot_waterfall(self, index: int = 0, output_dir: str = "reports/figures",
                       filename: Optional[str] = None):
        """Plot SHAP waterfall cho 1 sample."""
        if self.shap_values is None:
            raise RuntimeError("Call explain() first")

        os.makedirs(output_dir, exist_ok=True)
        fname = filename or f"shap_waterfall_sample_{index+1}.png"

        plt.figure(figsize=(10, 6))
        if self.explainer_type == "text":
            shap.plots.waterfall(self.shap_values[index, :, 'Depression'], max_display=15, show=False)
        else:
            shap.plots.waterfall(self.shap_values[index], max_display=15, show=False)

        plt.title(f'SHAP Waterfall — Sample {index+1}')
        plt.tight_layout()
        path = os.path.join(output_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP waterfall saved to {path}")
        return path
