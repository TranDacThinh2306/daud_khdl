"""
feature_selector.py - Feature selection/importance ranking
===========================================================
Select the most relevant features for depression detection.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    mutual_info_classif,
    f_classif,
)
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Feature selection and importance ranking."""

    def __init__(self, method: str = "mutual_info", k: int = 50):
        self.method = method
        self.k = k
        self.selector = None
        self.feature_importances_ = None
        self.selected_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> "FeatureSelector":
        """Fit the feature selector."""
        if self.method == "chi2":
            self.selector = SelectKBest(chi2, k=min(self.k, X.shape[1]))
        elif self.method == "mutual_info":
            self.selector = SelectKBest(mutual_info_classif, k=min(self.k, X.shape[1]))
        elif self.method == "f_classif":
            self.selector = SelectKBest(f_classif, k=min(self.k, X.shape[1]))
        elif self.method == "random_forest":
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            self.feature_importances_ = rf.feature_importances_
            top_indices = np.argsort(self.feature_importances_)[::-1][: self.k]
            if feature_names is not None:
                self.selected_features_ = [feature_names[i] for i in top_indices]
            return self
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.selector.fit(X, y)
        if feature_names is not None:
            mask = self.selector.get_support()
            self.selected_features_ = [f for f, m in zip(feature_names, mask) if m]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted selector."""
        if self.method == "random_forest":
            top_indices = np.argsort(self.feature_importances_)[::-1][: self.k]
            return X[:, top_indices]
        return self.selector.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> np.ndarray:
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_importance_ranking(self, feature_names: List[str]) -> pd.DataFrame:
        """Get ranked feature importances."""
        if self.feature_importances_ is not None:
            importances = self.feature_importances_
        elif self.selector is not None:
            importances = self.selector.scores_
        else:
            raise RuntimeError("Selector not fitted yet")

        ranking = pd.DataFrame({
            "feature": feature_names[:len(importances)],
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return ranking
