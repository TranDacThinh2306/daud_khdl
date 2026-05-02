"""
text_features.py - TF-IDF, BoW, N-grams, GloVe
=================================================
Text-based feature extraction for depression detection.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    """Extract text features using TF-IDF, BoW, N-grams, and word embeddings."""

    def __init__(
        self,
        method: str = "tfidf",
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        embedding_dim: int = 300,
        embedding_path: Optional[str] = None,
    ):
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.embedding_dim = embedding_dim
        self.embedding_path = embedding_path
        self.vectorizer = None
        self.embedding_model = None

    def _init_tfidf(self):
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
        )

    def _init_bow(self):
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )

    def _load_glove(self):
        """Load pre-trained GloVe embeddings."""
        if self.embedding_path is None:
            logger.warning("No GloVe path provided, using random embeddings")
            return
        logger.info(f"Loading GloVe embeddings from {self.embedding_path}")
        self.embedding_model = {}
        with open(self.embedding_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                self.embedding_model[word] = vector
        logger.info(f"Loaded {len(self.embedding_model)} word vectors")

    def _text_to_glove(self, text: str) -> np.ndarray:
        """Convert text to averaged GloVe vector."""
        words = text.split()
        vectors = []
        for word in words:
            if self.embedding_model and word in self.embedding_model:
                vectors.append(self.embedding_model[word])
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.embedding_dim)

    def fit(self, texts: pd.Series) -> "TextFeatureExtractor":
        """Fit the feature extractor on training texts."""
        if self.method in ("tfidf", "tf-idf"):
            self._init_tfidf()
            self.vectorizer.fit(texts)
        elif self.method == "bow":
            self._init_bow()
            self.vectorizer.fit(texts)
        elif self.method == "glove":
            self._load_glove()
        logger.info(f"TextFeatureExtractor fitted with method={self.method}")
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        """Transform texts to feature vectors."""
        if self.method in ("tfidf", "tf-idf", "bow"):
            features = self.vectorizer.transform(texts)
            if issparse(features):
                features = features.toarray()
            return features
        elif self.method == "glove":
            return np.vstack(texts.apply(self._text_to_glove).values)
        raise ValueError(f"Unknown method: {self.method}")

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self):
        """Get feature names (available for TF-IDF and BoW)."""
        if self.vectorizer and hasattr(self.vectorizer, "get_feature_names_out"):
            return self.vectorizer.get_feature_names_out()
        return None
