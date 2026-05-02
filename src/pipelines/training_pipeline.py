"""
training_pipeline.py - End-to-end training + XAI logging
==========================================================
Orchestrates data loading, feature extraction, model training, and XAI.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.data.preprocess import TextPreprocessor
from src.data.utils import load_dataset, split_dataset
from src.features.text_features import TextFeatureExtractor
from src.features.linguistic_features import LinguisticFeatureExtractor
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.models.registry import ModelRegistry
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.visualizer import ExplanationVisualizer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline with XAI integration."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.preprocessor = TextPreprocessor()
        self.text_extractor = TextFeatureExtractor(
            method=self.config.get("feature_method", "tfidf")
        )
        self.ling_extractor = LinguisticFeatureExtractor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.registry = ModelRegistry()
        self.visualizer = ExplanationVisualizer()

    def run(
        self,
        data_path: str,
        model: Any,
        experiment_id: str,
        generate_xai: bool = True,
    ) -> Dict:
        """
        Run the full training pipeline.

        Args:
            data_path: Path to raw data CSV
            model: Untrained model instance
            experiment_id: Unique experiment identifier
            generate_xai: Whether to generate XAI reports

        Returns:
            Dictionary with model, metrics, and explanations
        """
        # 1. Load and preprocess data
        logger.info("Step 1: Loading and preprocessing data...")
        df = load_dataset(data_path)
        df = self.preprocessor.process_dataframe(df)

        # 2. Extract features
        logger.info("Step 2: Extracting features...")
        text_features = self.text_extractor.fit_transform(df["cleaned_text"])
        ling_features = self.ling_extractor.extract_batch(df["cleaned_text"])
        X = np.hstack([text_features, ling_features.values])
        y = df["label"].values

        feature_names = list(self.text_extractor.get_feature_names() or [])
        feature_names += list(ling_features.columns)

        # 3. Split data
        logger.info("Step 3: Splitting data...")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 4. Train model
        logger.info("Step 4: Training model...")
        trained_model = self.trainer.train(model, X_train, y_train)

        # 5. Evaluate
        logger.info("Step 5: Evaluating model...")
        metrics = self.evaluator.evaluate(trained_model, X_test, y_test)

        # 6. Generate XAI explanations
        explainer = None
        if generate_xai:
            logger.info("Step 6: Generating XAI explanations...")
            shap_exp = SHAPExplainer(trained_model)
            shap_exp.fit(X_train)
            shap_values = shap_exp.explain(X_test[:100])
            self.visualizer.plot_shap_summary(
                shap_values, X_test[:100], feature_names
            )
            importance = shap_exp.get_feature_importance(X_test[:100], feature_names)
            self.visualizer.plot_feature_importance(importance)
            explainer = shap_exp

        # 7. Save experiment
        logger.info("Step 7: Saving experiment...")
        self.registry.save_experiment(
            trained_model, metrics, experiment_id, feature_names, explainer
        )

        return {
            "model": trained_model,
            "metrics": metrics,
            "feature_names": feature_names,
            "explainer": explainer,
        }
