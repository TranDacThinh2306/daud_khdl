"""
behavioral_features.py - Behavioral cues (screen time, nighttime usage)
========================================================================
Extract behavioral features from user activity metadata.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class BehavioralFeatureExtractor:
    """Extract behavioral features from user activity patterns."""

    def __init__(self, night_start: int = 23, night_end: int = 5):
        self.night_start = night_start
        self.night_end = night_end

    def _is_nighttime(self, hour: int) -> bool:
        return hour >= self.night_start or hour < self.night_end

    def extract_temporal(self, timestamps: pd.Series) -> pd.DataFrame:
        """Extract temporal behavioral features from timestamps."""
        ts = pd.to_datetime(timestamps, errors="coerce")
        features = pd.DataFrame()
        features["behav_hour"] = ts.dt.hour
        features["behav_day_of_week"] = ts.dt.dayofweek
        features["behav_is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        features["behav_is_nighttime"] = ts.dt.hour.apply(
            lambda h: 1 if self._is_nighttime(h) else 0
        )
        return features

    def extract_user_patterns(self, df: pd.DataFrame, user_col: str = "user_id",
                               timestamp_col: str = "timestamp") -> pd.DataFrame:
        """Extract per-user behavioral patterns."""
        df = df.copy()
        df["_ts"] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df["_hour"] = df["_ts"].dt.hour
        df["_is_night"] = df["_hour"].apply(lambda h: 1 if self._is_nighttime(h) else 0)

        user_features = df.groupby(user_col).agg(
            behav_post_count=("_ts", "count"),
            behav_night_ratio=("_is_night", "mean"),
            behav_avg_hour=("_hour", "mean"),
            behav_hour_std=("_hour", "std"),
        ).reset_index()

        user_features["behav_hour_std"] = user_features["behav_hour_std"].fillna(0)
        return user_features

    def extract_posting_frequency(self, df: pd.DataFrame, user_col: str = "user_id",
                                   timestamp_col: str = "timestamp") -> pd.DataFrame:
        """Calculate posting frequency features per user."""
        df = df.copy()
        df["_ts"] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.sort_values([user_col, "_ts"])

        df["_time_diff"] = df.groupby(user_col)["_ts"].diff().dt.total_seconds() / 3600
        freq = df.groupby(user_col).agg(
            behav_avg_time_between_posts=("_time_diff", "mean"),
            behav_min_time_between_posts=("_time_diff", "min"),
        ).reset_index()

        freq = freq.fillna(0)
        return freq
