"""
train_and_explain.py - Huấn luyện + xuất XAI reports
=======================================================
End-to-end script for training models and generating XAI reports.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_only(data_path: str, model_type: str = "random_forest"):
    """Train model without XAI reports."""
    from src.pipelines.training_pipeline import TrainingPipeline
    from src.models.architectures.random_forest import build_random_forest
    from src.models.architectures.svm import build_svm
    from src.models.architectures.xgboost import build_xgboost

    model_builders = {
        "random_forest": build_random_forest,
        "svm": build_svm,
        "xgboost": build_xgboost,
    }

    if model_type not in model_builders:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_builders.keys())}")

    model = model_builders[model_type]()
    experiment_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    pipeline = TrainingPipeline()
    results = pipeline.run(
        data_path=data_path,
        model=model,
        experiment_id=experiment_id,
        generate_xai=False,
    )

    logger.info(f"Training complete. Metrics: {results['metrics']}")
    return results


def train_and_explain(data_path: str, model_type: str = "random_forest"):
    """Train model and generate full XAI reports."""
    from src.pipelines.training_pipeline import TrainingPipeline
    from src.models.architectures.random_forest import build_random_forest
    from src.models.architectures.svm import build_svm
    from src.models.architectures.xgboost import build_xgboost

    model_builders = {
        "random_forest": build_random_forest,
        "svm": build_svm,
        "xgboost": build_xgboost,
    }

    model = model_builders.get(model_type, build_random_forest)()
    experiment_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    pipeline = TrainingPipeline()
    results = pipeline.run(
        data_path=data_path,
        model=model,
        experiment_id=experiment_id,
        generate_xai=True,
    )

    logger.info(f"Training + XAI complete. Metrics: {results['metrics']}")
    logger.info("XAI reports saved in reports/figures/")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train and explain depression detection models")
    parser.add_argument("--mode", choices=["train", "full"], default="full")
    parser.add_argument("--data", default="data/raw/comments_raw.csv")
    parser.add_argument("--model", default="random_forest",
                        choices=["random_forest", "svm", "xgboost"])
    args = parser.parse_args()

    if args.mode == "train":
        train_only(args.data, args.model)
    else:
        train_and_explain(args.data, args.model)


if __name__ == "__main__":
    main()
