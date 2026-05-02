"""
random_forest.py - RF (accuracy ~84% với behavioral features)
==============================================================
Random Forest classifier for depression detection.
"""

from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier


def build_random_forest(
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    class_weight: str = "balanced",
    random_state: int = 42,
    **kwargs,
) -> RandomForestClassifier:
    """
    Build a Random Forest classifier optimized for depression detection.

    Expected accuracy ~84% with behavioral features.

    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples per leaf
        class_weight: Class weighting strategy
        random_state: Random seed

    Returns:
        Configured RandomForestClassifier
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )


def get_param_grid() -> Dict:
    """Get hyperparameter grid for tuning."""
    return {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
