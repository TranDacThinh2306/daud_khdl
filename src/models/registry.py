"""
registry.py - Model registry cho versioning
=============================================
Model versioning, saving, and loading.
"""

import os
import json
import logging
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing model versions and artifacts."""

    def __init__(self, base_dir: str = "models_saved"):
        self.base_dir = base_dir
        self.experiments_dir = os.path.join(base_dir, "experiments")
        self.production_dir = os.path.join(base_dir, "production")
        os.makedirs(self.experiments_dir, exist_ok=True)
        os.makedirs(self.production_dir, exist_ok=True)

    def save_experiment(
        self,
        model: Any,
        metrics: Dict,
        experiment_id: str,
        feature_names: Optional[list] = None,
        explainer: Optional[Any] = None,
    ) -> str:
        """Save model and artifacts for an experiment."""
        exp_dir = os.path.join(self.experiments_dir, f"experiment_{experiment_id}")
        os.makedirs(exp_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(exp_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metrics
        metrics_path = os.path.join(exp_dir, "metrics.json")
        serializable = {k: float(v) if v is not None else None for k, v in metrics.items()}
        serializable["saved_at"] = datetime.now().isoformat()
        serializable["model_class"] = model.__class__.__name__
        with open(metrics_path, "w") as f:
            json.dump(serializable, f, indent=2)

        # Save feature names
        if feature_names:
            fn_path = os.path.join(exp_dir, "feature_names.pkl")
            with open(fn_path, "wb") as f:
                pickle.dump(feature_names, f)

        # Save SHAP explainer
        if explainer:
            exp_path = os.path.join(exp_dir, "shap_explainer.pkl")
            with open(exp_path, "wb") as f:
                pickle.dump(explainer, f)

        logger.info(f"Experiment {experiment_id} saved to {exp_dir}")
        return exp_dir

    def load_experiment(self, experiment_id: str) -> Dict:
        """Load a saved experiment."""
        exp_dir = os.path.join(self.experiments_dir, f"experiment_{experiment_id}")
        if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"Experiment {experiment_id} not found")

        result = {}
        model_path = os.path.join(exp_dir, "model.pkl")
        with open(model_path, "rb") as f:
            result["model"] = pickle.load(f)

        metrics_path = os.path.join(exp_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                result["metrics"] = json.load(f)

        return result

    def promote_to_production(self, experiment_id: str) -> str:
        """Promote an experiment model to production."""
        exp = self.load_experiment(experiment_id)
        prod_path = os.path.join(self.production_dir, "best_model.pkl")
        with open(prod_path, "wb") as f:
            pickle.dump(exp["model"], f)
        logger.info(f"Experiment {experiment_id} promoted to production")
        return prod_path

    def load_production_model(self) -> Any:
        """Load the current production model."""
        prod_path = os.path.join(self.production_dir, "best_model.pkl")
        if not os.path.exists(prod_path):
            raise FileNotFoundError("No production model found")
        with open(prod_path, "rb") as f:
            return pickle.load(f)
