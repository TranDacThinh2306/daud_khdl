"""
augment.py - Data augmentation cho dữ liệu ít
================================================
Data augmentation techniques for handling imbalanced depression datasets.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataAugmenter:
    """Data augmentation for text classification with imbalanced classes."""

    def __init__(self, random_state: int = 42):
        """
        Initialize the augmenter.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        Replace n random words with their synonyms using nlpaug.

        Args:
            text: Input text
            n: Number of words to replace

        Returns:
            Augmented text
        """
        try:
            import nlpaug.augmenter.word as naw

            aug = naw.SynonymAug(
                aug_src="wordnet",
                aug_min=n,
                aug_max=n,
            )
            return aug.augment(text)[0] if isinstance(aug.augment(text), list) else aug.augment(text)
        except Exception as e:
            logger.warning(f"Synonym replacement failed: {e}")
            return text

    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert synonyms into the text.

        Args:
            text: Input text
            n: Number of insertions

        Returns:
            Augmented text
        """
        try:
            import nlpaug.augmenter.word as naw

            aug = naw.SynonymAug(
                aug_src="wordnet",
                action="insert",
                aug_min=n,
                aug_max=n,
            )
            return aug.augment(text)[0] if isinstance(aug.augment(text), list) else aug.augment(text)
        except Exception as e:
            logger.warning(f"Random insertion failed: {e}")
            return text

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words with probability p.

        Args:
            text: Input text
            p: Probability of deleting each word

        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) <= 1:
            return text

        remaining = [w for w in words if np.random.random() > p]
        if len(remaining) == 0:
            return words[np.random.randint(0, len(words))]
        return " ".join(remaining)

    def back_translation(self, text: str, src_lang: str = "en", pivot_lang: str = "de") -> str:
        """
        Augment text via back-translation (en -> pivot -> en).

        Args:
            text: Input text
            src_lang: Source language
            pivot_lang: Pivot language for translation

        Returns:
            Back-translated text
        """
        try:
            import nlpaug.augmenter.word as naw

            aug = naw.BackTranslationAug(
                from_model_name=f"Helsinki-NLP/opus-mt-{src_lang}-{pivot_lang}",
                to_model_name=f"Helsinki-NLP/opus-mt-{pivot_lang}-{src_lang}",
            )
            return aug.augment(text)[0] if isinstance(aug.augment(text), list) else aug.augment(text)
        except Exception as e:
            logger.warning(f"Back translation failed: {e}")
            return text

    def oversample_minority(
        self,
        df: pd.DataFrame,
        text_column: str = "cleaned_text",
        label_column: str = "label",
        target_ratio: float = 1.0,
        methods: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Oversample the minority class using text augmentation.

        Args:
            df: Input DataFrame
            text_column: Column containing text
            label_column: Column containing labels
            target_ratio: Desired ratio of minority to majority class
            methods: List of augmentation methods to use

        Returns:
            Augmented DataFrame with balanced classes
        """
        if methods is None:
            methods = ["synonym", "deletion"]

        label_counts = df[label_column].value_counts()
        majority_label = label_counts.idxmax()
        minority_label = label_counts.idxmin()

        majority_count = label_counts[majority_label]
        minority_count = label_counts[minority_label]
        target_count = int(majority_count * target_ratio)
        samples_needed = target_count - minority_count

        if samples_needed <= 0:
            logger.info("Classes already balanced, no augmentation needed")
            return df

        logger.info(
            f"Augmenting {minority_label} class: {minority_count} -> {target_count} "
            f"(+{samples_needed} samples)"
        )

        minority_df = df[df[label_column] == minority_label]
        augmented_rows = []

        for _ in range(samples_needed):
            row = minority_df.sample(1, random_state=self.random_state).iloc[0].copy()
            method = np.random.choice(methods)

            if method == "synonym":
                row[text_column] = self.synonym_replacement(row[text_column])
            elif method == "insertion":
                row[text_column] = self.random_insertion(row[text_column])
            elif method == "deletion":
                row[text_column] = self.random_deletion(row[text_column])
            elif method == "back_translation":
                row[text_column] = self.back_translation(row[text_column])

            augmented_rows.append(row)

        augmented_df = pd.DataFrame(augmented_rows)
        result = pd.concat([df, augmented_df], ignore_index=True)
        logger.info(f"Augmentation complete. Total samples: {len(result)}")
        return result
