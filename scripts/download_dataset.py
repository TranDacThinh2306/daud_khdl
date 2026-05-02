"""
download_dataset.py - Tải eRisk, SMHD datasets
==================================================
Script to download and prepare standard depression datasets.
"""

import os
import sys
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_erisk():
    """Download eRisk depression dataset (requires registration)."""
    logger.info("eRisk dataset requires manual registration at:")
    logger.info("  https://erisk.irlab.org/")
    logger.info("Please download manually and place in data/external/erisk/")


def download_smhd():
    """Download SMHD dataset (requires data use agreement)."""
    logger.info("SMHD dataset requires a Data Use Agreement.")
    logger.info("  Paper: https://aclanthology.org/C18-1126/")
    logger.info("Please download manually and place in data/external/smhd/")


def download_sample_data():
    """Download a sample dataset for testing."""
    import pandas as pd
    import numpy as np

    logger.info("Generating sample dataset for testing...")
    np.random.seed(42)
    n = 1000

    depressed_phrases = [
        "I feel so empty inside",
        "nothing matters anymore",
        "I can't stop crying",
        "everything feels hopeless",
        "I just want to disappear",
        "I'm so tired of everything",
        "nobody understands me",
        "I feel worthless",
    ]
    normal_phrases = [
        "Had a great day today",
        "Love spending time with friends",
        "Just finished a good book",
        "Beautiful weather outside",
        "Excited about the weekend",
        "Feeling grateful today",
        "Made progress on my project",
        "Enjoyed a nice meal",
    ]

    texts, labels = [], []
    for _ in range(n):
        if np.random.random() < 0.3:
            texts.append(np.random.choice(depressed_phrases))
            labels.append(1)
        else:
            texts.append(np.random.choice(normal_phrases))
            labels.append(0)

    df = pd.DataFrame({
        "comment_id": range(n),
        "user_id": [f"user_{i % 100}" for i in range(n)],
        "text": texts,
        "platform": "sample",
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h").astype(str).tolist(),
        "label": labels,
    })

    output_path = "data/raw/comments_raw.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Sample dataset saved to {output_path} ({len(df)} records)")


def main():
    parser = argparse.ArgumentParser(description="Download depression detection datasets")
    parser.add_argument(
        "--dataset",
        choices=["erisk", "smhd", "sample", "all"],
        default="sample",
        help="Dataset to download",
    )
    args = parser.parse_args()

    if args.dataset in ("erisk", "all"):
        download_erisk()
    if args.dataset in ("smhd", "all"):
        download_smhd()
    if args.dataset in ("sample", "all"):
        download_sample_data()


if __name__ == "__main__":
    main()
