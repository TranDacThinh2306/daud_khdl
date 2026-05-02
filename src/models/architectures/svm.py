"""
svm.py - SVM (tốt nhất cho text features)
===========================================
Support Vector Machine classifier for text-based depression detection.
"""

from typing import Dict
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_svm(
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    class_weight: str = "balanced",
    probability: bool = True,
    random_state: int = 42,
    **kwargs,
) -> Pipeline:
    """
    Build an SVM classifier pipeline with scaling.

    Best suited for text-based features (TF-IDF, BoW).

    Returns:
        Pipeline with StandardScaler + SVC
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            probability=probability,
            random_state=random_state,
            **kwargs,
        )),
    ])


def get_param_grid() -> Dict:
    """Get hyperparameter grid for tuning."""
    return {
        "svm__C": [0.1, 1.0, 10.0, 100.0],
        "svm__kernel": ["rbf", "linear"],
        "svm__gamma": ["scale", "auto"],
    }
