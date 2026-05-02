"""
preprocess.py - Làm sạch text (emoji, slang, stopwords)
========================================================
Text preprocessing pipeline for cleaning social media comments.
"""

import os
import re
import logging
import warnings
from typing import List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)



class TextPreprocessor:
    """Preprocesses text data for depression detection."""

    # Common slang/abbreviation mappings
    SLANG_MAP = {
        "brb": "be right back",
        "btw": "by the way",
        "idk": "i do not know",
        "imo": "in my opinion",
        "tbh": "to be honest",
        "smh": "shaking my head",
        "ngl": "not gonna lie",
        "irl": "in real life",
        "fomo": "fear of missing out",
        "tfw": "that feeling when",
        "mfw": "my face when",
        "af": "as fuck",
        "lmao": "laughing my ass off",
        "lol": "laughing out loud",
        "omg": "oh my god",
        "pls": "please",
        "thx": "thanks",
        "u": "you",
        "ur": "your",
        "r": "are",
        "n": "and",
        "bc": "because",
        "w/": "with",
        "w/o": "without",
    }

    # Emoji sentiment patterns
    POSITIVE_EMOJIS = re.compile(r"[😀😁😂🤣😃😄😅😆😉😊😋😎😍🥰😘😗😙😚🤗🤩🥳💪👍❤️💕💖]")
    NEGATIVE_EMOJIS = re.compile(r"[😢😭😞😔😟😕🙁😣😖😩😫😤😠😡💔😿🥺😰😥😓]")

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        remove_emojis: bool = False,
        expand_slang: bool = True,
        remove_stopwords: bool = False,
        min_length: int = 3,
        language: str = "english",
    ):
        """
        Initialize the preprocessor.

        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_mentions: Remove @mentions
            remove_hashtags: Remove hashtag symbols (keeps text)
            remove_emojis: Remove emoji characters
            expand_slang: Expand common slang/abbreviations
            remove_stopwords: Remove stopwords
            min_length: Minimum word length to keep
            language: Language for stopwords
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_emojis = remove_emojis
        self.expand_slang = expand_slang
        self.remove_stopwords = remove_stopwords
        self.min_length = min_length
        self.language = language
        self._stopwords = None

    def _get_stopwords(self) -> set:
        """Load stopwords lazily."""
        if self._stopwords is None:
            try:
                from nltk.corpus import stopwords

                self._stopwords = set(stopwords.words(self.language))
            except ImportError:
                logger.warning("NLTK not available, using empty stopwords set")
                self._stopwords = set()
        return self._stopwords

    def extract_emoji_features(self, text: str) -> dict:
        """Extract emoji-based features before cleaning."""
        return {
            "positive_emoji_count": len(self.POSITIVE_EMOJIS.findall(text)),
            "negative_emoji_count": len(self.NEGATIVE_EMOJIS.findall(text)),
            "total_emoji_count": len(re.findall(r"[\U00010000-\U0010ffff]", text)),
        }

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text string
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove mentions
        if self.remove_mentions:
            text = re.sub(r"@\w+", "", text)

        # Remove hashtag symbols (keep the text)
        if self.remove_hashtags:
            text = re.sub(r"#", "", text)

        # Remove emojis
        if self.remove_emojis:
            text = re.sub(r"[\U00010000-\U0010ffff]", "", text)

        # Expand slang
        if self.expand_slang:
            words = text.split()
            words = [self.SLANG_MAP.get(w.lower(), w) for w in words]
            text = " ".join(words)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove special characters (keep basic punctuation)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove stopwords
        if self.remove_stopwords:
            stop = self._get_stopwords()
            words = text.split()
            words = [w for w in words if w not in stop and len(w) >= self.min_length]
            text = " ".join(words)

        return text

    def process_list_texts(self, texts: List[str]) -> List[str]:
        """
        Process a list of text strings.

        Args:
            texts: List of raw text strings

        Returns:
            List of cleaned text strings
        """
        return [self.clean_text(text) for text in texts]
        
    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        output_column: str = "cleaned_text",
        emoji_processed: bool = False,   
        log_path: str = 'data/logs/cleaned_records_log.csv'
    ) -> pd.DataFrame:
        """
        Process an entire DataFrame of comments.

        Args:
            df: Input DataFrame with raw text
            text_column: Name of the text column
            output_column: Name for the cleaned text column
            emoji_processed: Whether to extract emoji features

        Returns:
            DataFrame with cleaned text added
        """
        logger.info(f"Processing {len(df)} comments...")

        # Extract emoji features before cleaning
        if emoji_processed:
            emoji_features = df[text_column].apply(self.extract_emoji_features)
            emoji_df = pd.DataFrame(emoji_features.tolist())
            df = pd.concat([df, emoji_df], axis=1)

        # Clean text
        df[output_column] = df[text_column].apply(self.clean_text)

        # Lưu log các record bị thay đổi sau khi clean
        changed_mask = df[text_column] != df[output_column]
        changed_df = df[changed_mask][[text_column, output_column]]
        if not changed_df.empty and log_path:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            changed_df.to_csv(log_path, index=False, encoding='utf-8')
            logger.info(f"Đã lưu {len(changed_df)} bản ghi có thay đổi vào {log_path}")

        # Remove empty texts
        initial_count = len(df)
        df = df[df[output_column].str.len() > 0].reset_index(drop=True)
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} empty comments after cleaning")

        logger.info(f"Processing complete. {len(df)} comments remaining.")
        return df

    def save_processed(self, df: pd.DataFrame, output_path: str):
        """Save processed DataFrame to CSV."""
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Saved processed data to {output_path}")
