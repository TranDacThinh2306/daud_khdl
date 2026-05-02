"""
linguistic_features.py - Linguistic markers (LIWC-style)
=========================================================
Extract linguistic features indicative of depression.
"""

import re
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Depression-related word categories (LIWC-inspired)
DEPRESSION_LEXICON = {
    "sadness": ["sad", "unhappy", "miserable", "depressed", "hopeless", "empty", "numb", "crying"],
    "anxiety": ["anxious", "worried", "nervous", "panic", "fear", "scared", "stress", "overwhelmed"],
    "anger": ["angry", "furious", "hate", "rage", "irritated", "frustrated", "mad", "annoyed"],
    "death": ["die", "death", "suicide", "kill", "dead", "funeral", "grave", "end it"],
    "loneliness": ["alone", "lonely", "isolated", "nobody", "no one", "abandoned", "rejected"],
    "negation": ["no", "not", "never", "nothing", "nowhere", "neither", "nobody", "none"],
    "first_person": ["i", "me", "my", "mine", "myself"],
    "absolutist": ["always", "never", "completely", "nothing", "everything", "totally", "absolutely"],
    "cognitive": ["think", "know", "believe", "understand", "realize", "remember", "forget"],
    "somatic": ["tired", "exhausted", "sleep", "insomnia", "pain", "headache", "sick", "fatigue"],
}


class LinguisticFeatureExtractor:
    """Extract LIWC-style linguistic features from text."""

    def __init__(self, lexicon: Dict[str, List[str]] = None):
        self.lexicon = lexicon or DEPRESSION_LEXICON

    def _count_category(self, words: List[str], category: str) -> int:
        category_words = set(self.lexicon.get(category, []))
        return sum(1 for w in words if w in category_words)

    def extract_single(self, text: str) -> Dict[str, float]:
        """Extract linguistic features from a single text."""
        words = text.lower().split()
        n_words = max(len(words), 1)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        n_sentences = max(len(sentences), 1)

        features = {}

        # Category word ratios
        for category in self.lexicon:
            count = self._count_category(words, category)
            features[f"ling_{category}_count"] = count
            features[f"ling_{category}_ratio"] = count / n_words

        # Structural features
        features["ling_word_count"] = n_words
        features["ling_avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
        features["ling_sentence_count"] = n_sentences
        features["ling_avg_sentence_length"] = n_words / n_sentences
        features["ling_exclamation_count"] = text.count("!")
        features["ling_question_count"] = text.count("?")
        features["ling_ellipsis_count"] = text.count("...")
        features["ling_caps_ratio"] = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        # Vocabulary richness (type-token ratio)
        features["ling_ttr"] = len(set(words)) / n_words

        return features

    def extract_batch(self, texts: pd.Series) -> pd.DataFrame:
        """Extract linguistic features from a batch of texts."""
        logger.info(f"Extracting linguistic features from {len(texts)} texts...")
        features = texts.apply(self.extract_single)
        return pd.DataFrame(features.tolist())
