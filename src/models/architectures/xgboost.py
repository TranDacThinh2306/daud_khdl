"""
xgboost.py - XGBoost cho ensemble
====================================
XGBoost gradient boosted classifier for depression detection.
"""

from typing import Dict


def build_xgboost(
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
    **kwargs,
):
    """
    Build an XGBoost classifier for ensemble learning.

    Returns:
        Configured XGBClassifier
    """
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
        **kwargs,
    )


def get_param_grid() -> Dict:
    """Get hyperparameter grid for tuning."""
    return {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.7, 0.8, 0.9],
    }
