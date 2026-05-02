"""
Model architectures for depression classification.
"""

from .random_forest import build_random_forest
from .svm import build_svm
from .xgboost import build_xgboost

__all__ = ["build_random_forest", "build_svm", "build_xgboost"]
