"""
text_utils.py - Text normalization, tokenization
===================================================
Common text processing utilities.
"""

import re
import unicodedata
from typing import List, Optional


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters."""
    return unicodedata.normalize("NFKD", text)


def remove_repeated_chars(text: str, max_repeat: int = 3) -> str:
    """Reduce repeated characters (e.g., 'soooo' -> 'sooo')."""
    pattern = r"(.)\1{" + str(max_repeat) + r",}"
    return re.sub(pattern, r"\1" * max_repeat, text)


def tokenize_simple(text: str) -> List[str]:
    """Simple whitespace tokenization with basic cleaning."""
    text = re.sub(r"[^\w\s]", " ", text)
    return text.lower().split()


def tokenize_with_spacy(text: str, model_name: str = "en_core_web_sm") -> List[str]:
    """Tokenize using spaCy."""
    import spacy
    nlp = spacy.load(model_name)
    doc = nlp(text)
    return [token.text.lower() for token in doc if not token.is_space]


def lemmatize(text: str, model_name: str = "en_core_web_sm") -> str:
    """Lemmatize text using spaCy."""
    import spacy
    nlp = spacy.load(model_name)
    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc if not token.is_space])


def detect_negation(text: str) -> bool:
    """Detect if text contains negation patterns."""
    negation_patterns = [
        r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bnothing\b",
        r"\bnowhere\b", r"\bneither\b", r"\bnobody\b",
        r"\bcan't\b", r"\bwon't\b", r"\bdon't\b", r"\bdoesn't\b",
        r"\bisn't\b", r"\baren't\b", r"\bwasn't\b", r"\bweren't\b",
    ]
    for pattern in negation_patterns:
        if re.search(pattern, text.lower()):
            return True
    return False


def count_sentences(text: str) -> int:
    """Count number of sentences in text."""
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])
