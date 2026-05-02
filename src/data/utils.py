"""
utils.py - Data utility functions
==================================
Helper functions for data loading, splitting, and validation.
"""

import os
import logging
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_dataset(
    filepath: str,
    text_column: str = "text",
    label_column: str = "label",
) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Args:
        filepath: Path to CSV file
        text_column: Name of the text column
        label_column: Name of the label column

    Returns:
        Loaded DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"Loaded {len(df)} records from {filepath}")

    # Validate required columns
    for col in [text_column, label_column]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset")

    # Log class distribution
    dist = df[label_column].value_counts()
    logger.info(f"Class distribution:\n{dist}")

    return df


def split_dataset(
    df: pd.DataFrame,
    label_column: str = "label",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        df: Input DataFrame
        label_column: Label column for stratification
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        stratify: Whether to use stratified splitting

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    strat = df[label_column] if stratify else None

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat
    )

    strat_train = train_df[label_column] if stratify else None
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_ratio, random_state=random_state, stratify=strat_train
    )

    logger.info(
        f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    return train_df, val_df, test_df


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """Validate that a DataFrame contains required columns."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def get_class_weights(labels: np.ndarray) -> dict:
    """Calculate class weights for imbalanced datasets."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes, weights))
