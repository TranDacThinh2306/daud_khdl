"""
test_data.py - Tests for data pipeline
========================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data.preprocess import TextPreprocessor
from src.data.augment import DataAugmenter
from src.data.load_dataset import get_dataset
from src.data.dataset import DepressionDataset, DataLoader
from transformers import AutoTokenizer

class TestTextPreprocessor:
    """Tests for TextPreprocessor."""

    def setup_method(self):
        self.preprocessor = TextPreprocessor()

    def test_clean_text_removes_urls(self):
        text = "Check this https://example.com for more info"
        cleaned = self.preprocessor.clean_text(text)
        assert "https" not in cleaned
        assert "example.com" not in cleaned

    def test_clean_text_removes_mentions(self):
        text = "@user1 I feel so sad today"
        cleaned = self.preprocessor.clean_text(text)
        assert "@user1" not in cleaned

    def test_clean_text_lowercase(self):
        text = "I Am VERY Sad"
        cleaned = self.preprocessor.clean_text(text)
        assert cleaned == cleaned.lower()

    def test_clean_text_empty_string(self):
        assert self.preprocessor.clean_text("") == ""
        assert self.preprocessor.clean_text("   ") == ""

    def test_expand_slang(self):
        preprocessor = TextPreprocessor(expand_slang=True, lowercase=False)
        text = "idk what to do tbh"
        cleaned = preprocessor.clean_text(text)
        assert "i do not know" in cleaned

    def test_extract_emoji_features(self):
        text = "I'm so sad 😢😭"
        features = self.preprocessor.extract_emoji_features(text)
        assert features["negative_emoji_count"] >= 2

    def test_process_dataframe(self):
        df = pd.DataFrame({
            "text": ["Hello world!", "I feel sad today", ""],
            "label": [0, 1, 0],
        })
        result = self.preprocessor.process_dataframe(df)
        assert "cleaned_text" in result.columns
        assert len(result) <= len(df)
    
    def test_function_dataset_process(self):
        train_df, test_df = get_dataset('hugginglearners/reddit-depression-cleaned')
        train_df = self.preprocessor.process_dataframe(df = train_df, text_column = 'clean_text', output_column = 'cleaned_text', emoji_processed = True)
        test_df = self.preprocessor.process_dataframe(df = test_df, text_column = 'clean_text', output_column = 'cleaned_text', emoji_processed = True)
        assert "cleaned_text" in train_df.columns
        assert "cleaned_text" in test_df.columns
        assert len(train_df) <= len(train_df)
        assert len(test_df) <= len(test_df)
        
        # self.visualize_dataset(train_df)
        # self.visualize_dataset(test_df)

    def test_function_processor(self):
        train_df, _ = get_dataset('hugginglearners/reddit-depression-cleaned')
        # self.visualize_dataset(train_df, text_column = 'clean_text')
        train_df = self.preprocessor.process_dataframe(df = train_df, text_column = 'clean_text', output_column = 'cleaned_text', emoji_processed = True)
        assert "cleaned_text" in train_df.columns
        assert len(train_df) <= len(train_df)
        # self.visualize_dataset(train_df)

    def test_function_dataset_class(self):
        berttokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        train_df, test_df = get_dataset('hugginglearners/reddit-depression-cleaned')
        train_df = self.preprocessor.process_dataframe(df = train_df, text_column = 'clean_text', output_column = 'cleaned_text', emoji_processed = True)
        test_df = self.preprocessor.process_dataframe(df = test_df, text_column = 'clean_text', output_column = 'cleaned_text', emoji_processed = True)
        train_dataset = DepressionDataset(train_df['cleaned_text'], train_df['is_depression'], tokenizer = berttokenizer)
        test_dataset = DepressionDataset(test_df['cleaned_text'], test_df['is_depression'], tokenizer = berttokenizer)
        assert len(train_dataset) <= len(train_df)
        assert len(test_dataset) <= len(test_df)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=16)
        
        # Get first batch and print 5 samples
        batch = next(iter(train_loader))
        print("\n--- 5 samples from train_loader ---")
        for i in range(5):
            print(f"Sample {i+1}:")
            print(f"  Input IDs shape: {batch['input_ids'][i].shape}")
            print(f"  Input IDs (first 10): {batch['input_ids'][i][:10].tolist()}...")
            print(f"  Attention Mask shape: {batch['attention_mask'][i].shape}")
            print(f"  Attention Mask (first 10): {batch['attention_mask'][i][:10].tolist()}...")
            print(f"  Label: {batch['label'][i].item()}")
            decoded_text = berttokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
            print(f"  Decoded text: {decoded_text[:150]}...")
            print("-" * 50)

    def visualize_dataset(self, df: pd.DataFrame, label_column: str = 'is_depression', text_column: str = 'cleaned_text'):
        # Phân phối nhãn
        print('--- Phân phối nhãn ---')
        print(df[label_column].value_counts())
        print(f'\nTỷ lệ depression: {df[label_column].mean():.1%}')

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        # --- Biểu đồ 1: Label distribution ---
        label_counts = df[label_column].value_counts().sort_index()
        axes[0].bar(['Non-depression', 'Depression'],
                    label_counts.values,
                    color=['#3B8BD4', '#D85A30'], alpha=0.85)
        axes[0].set_title('Phân phối nhãn')
        axes[0].set_ylabel('Số lượng bài')

        # --- Biểu đồ 2 & 3: Tách riêng từng nhóm ---
        df['text_len'] = df[text_column].str.split().str.len()

        for ax, label, color, name in zip(
            [axes[1], axes[2]],
            [0, 1],
            ['#3B8BD4', '#D85A30'],
            ['Non-depression', 'Depression']
        ):
            subset = df[df[label_column] == label]['text_len'].dropna()
            ax.hist(subset, bins=50, color=color, alpha=0.85)
            ax.set_title(f'Độ dài bài — {name}')
            ax.set_xlabel('Số từ')
            ax.set_ylabel('Số bài')

            # Thêm thống kê
            if not subset.empty:
                median_val = subset.median()
                ax.axvline(median_val, color='black', linestyle='--', linewidth=1)
                ax.text(median_val + 20, ax.get_ylim()[1] * 0.9,
                        f'median={median_val:.0f}', fontsize=9)

        plt.tight_layout()
        plt.show()

        # In thống kê tóm tắt
        print('--- Thống kê độ dài bài đăng ---')
        for label, name in [(0, 'Non-depression'), (1, 'Depression')]:
            s = df[df[label_column] == label]['text_len'].dropna()
            if not s.empty:
                print(f'{name}: median={s.median():.0f} | mean={s.mean():.0f} | max={s.max()}')


class TestDataAugmenter:
    """Tests for DataAugmenter."""

    def test_random_deletion(self):
        augmenter = DataAugmenter(random_state=42)
        text = "I am feeling very sad and lonely today"
        result = augmenter.random_deletion(text, p=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_single_word_not_deleted(self):
        augmenter = DataAugmenter()
        result = augmenter.random_deletion("hello", p=0.9)
        assert result == "hello"

if __name__ == "__main__":
    txtproc = TestTextPreprocessor()
    txtproc.setup_method()
    txtproc.test_function_dataset_class()