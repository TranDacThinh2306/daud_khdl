"""
scripts/run_explain.py - Giải thích dự đoán bằng SHAP & LIME
================================================================
Load mô hình DistilBERT đã train → chạy SHAP (global) + LIME (per-sample)
→ lưu kết quả giải thích vào reports/figures/

SHAP: Cung cấp insight cho TOÀN BỘ dataset (global feature importance)
LIME: Giải thích CHI TIẾT cho TỪNG mẫu riêng lẻ
"""

import sys
import os
import logging
import argparse
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Đảm bảo import từ root project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocess import TextPreprocessor
from src.data.load_dataset import get_dataset
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.visualizer import ExplanationVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Explain DistilBERT Depression Predictions (SHAP + LIME)')
    parser.add_argument('--model_path', type=str, default='models_saved/experiments',
                        help='Path to saved model directory')
    parser.add_argument('--output_dir', type=str, default='reports/figures',
                        help='Directory to save explanation figures')
    parser.add_argument('--num_samples', type=str, default='50',
                        help='Number of dataset samples ("all" or integer)')
    parser.add_argument('--shap_max_evals', type=int, default=500,
                        help='Max evaluations for SHAP Partition explainer')
    parser.add_argument('--lime_num_features', type=int, default=10,
                        help='Number of top features for LIME')
    parser.add_argument('--lime_num_samples', type=int, default=2000,
                        help='Number of perturbation samples for LIME')
    parser.add_argument('--dataset', type=str, default='hugginglearners/reddit-depression-cleaned',
                        help='HuggingFace dataset path')
    return parser.parse_args()


# ══════════════════════════════════════════════════
# Helper: Wrapper predict function cho transformer
# ══════════════════════════════════════════════════
class TransformerPredictor:
    """Wrapper để chuyển model transformer thành predict function nhận text."""

    def __init__(self, model, tokenizer, device, max_len=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.class_names = ['Non-depression', 'Depression']

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Nhận danh sách text → trả về probabilities [n_samples, 2]."""
        self.model.eval()
        all_probs = []

        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = enc['input_ids'].to(self.device)
            attn_mask = enc['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    def __call__(self, texts):
        """Cho SHAP gọi trực tiếp."""
        if isinstance(texts, str):
            texts = [texts]
        return self.predict_proba(list(texts))

    def get_prediction_info(self, text: str) -> str:
        """Trả về string mô tả prediction cho 1 sample."""
        probs = self.predict_proba([text])[0]
        pred_label = int(probs.argmax())
        return f"Pred: {self.class_names[pred_label]}, P={probs[pred_label]:.3f}"


# ══════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════
def run_explain_pipeline(args):
    """
    Pipeline giải thích:
      1. Load model
      2. Chuẩn bị texts
      3. SHAP — Global explanation cho toàn bộ dataset
      4. LIME — Local explanation cho từng mẫu
      5. So sánh SHAP vs LIME
    """

    # ── Step 1: Load model đã train ──
    logger.info("Step 1: Loading saved model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()
    print(f"  Model loaded from: {args.model_path}")
    print(f"  Device: {device}")

    predictor = TransformerPredictor(model, tokenizer, device)

    # ── Step 2: Chuẩn bị texts từ dataset ──
    logger.info("Step 2: Preparing sample texts...")
    texts = []
    preprocessor = TextPreprocessor()
    train_df, _ = get_dataset(args.dataset)
    train_df = preprocessor.process_dataframe(
        df=train_df, text_column='clean_text',
        output_column='cleaned_text', emoji_processed=True
    )

    if args.num_samples == 'all':
        logger.info("  Using ALL samples from dataset...")
        dep_samples = train_df[train_df['is_depression'] == 1]['cleaned_text'].tolist()
        non_dep_samples = train_df[train_df['is_depression'] == 0]['cleaned_text'].tolist()
        texts.extend(dep_samples + non_dep_samples)
        logger.info(f"  Added {len(dep_samples)} depression + {len(non_dep_samples)} non-depression samples")
    else:
        num_samples = int(args.num_samples)
        logger.info(f"  Loading {num_samples} real samples from dataset (balanced)...")
        n_each = num_samples // 2
        dep_samples = train_df[train_df['is_depression'] == 1]['cleaned_text'].sample(
            n=min(n_each, len(train_df[train_df['is_depression'] == 1])), random_state=42
        ).tolist()
        non_dep_samples = train_df[train_df['is_depression'] == 0]['cleaned_text'].sample(
            n=min(num_samples - n_each, len(train_df[train_df['is_depression'] == 0])), random_state=42
        ).tolist()
        texts.extend(dep_samples + non_dep_samples)
        logger.info(f"  Added {len(dep_samples)} depression + {len(non_dep_samples)} non-depression samples")

    print(f"  Total texts to explain: {len(texts)}")

    # Quick predict tất cả
    probs = predictor.predict_proba(texts)
    for i, (text, prob) in enumerate(zip(texts, probs)):
        pred = predictor.class_names[prob.argmax()]
        print(f"  [{i+1}] {pred} (P={prob.max():.3f}): {text[:80]}...")

    # ══════════════════════════════════════════════════
    # Step 3: SHAP — Global explanation cho toàn bộ dataset
    # ══════════════════════════════════════════════════
    logger.info("Step 3: SHAP — Global explanation for entire dataset...")
    shap_dir = os.path.join(args.output_dir, 'shap')

    # Khởi tạo SHAPExplainer với text mode
    shap_exp = SHAPExplainer(model=predictor, explainer_type="text")
    shap_exp.fit_text(
        predict_fn=predictor,
        tokenizer=tokenizer,
        output_names=predictor.class_names
    )

    # Explain toàn bộ texts → SHAP values cho dataset
    logger.info(f"  Running SHAP on {len(texts)} samples (max_evals={args.shap_max_evals})...")
    shap_values = shap_exp.explain(texts, max_evals=args.shap_max_evals)

    # 3a. Global bar plot — Top features quan trọng nhất trên TOÀN dataset
    shap_exp.plot_global_summary(output_dir=shap_dir, top_k=20)
    print(f"  ✓ SHAP global importance plot saved")

    # 3b. Global feature importance dictionary
    importance = shap_exp.get_feature_importance(texts, max_evals=args.shap_max_evals)
    top_20 = list(importance.items())[:20]
    print(f"\n  ── SHAP Global Top-20 Features (Depression) ──")
    for rank, (token, score) in enumerate(top_20, 1):
        print(f"    {rank:2d}. {token:20s} → {score:.4f}")

    # 3c. Waterfall cho vài sample minh họa
    for idx in range(min(3, len(texts))):
        shap_exp.plot_waterfall(index=idx, output_dir=shap_dir)
    print(f"  ✓ SHAP waterfall plots saved")

    # ══════════════════════════════════════════════════
    # Step 4: LIME — Local explanation cho TỪNG mẫu
    # ══════════════════════════════════════════════════
    logger.info("Step 4: LIME — Local explanation per sample...")
    lime_dir = os.path.join(args.output_dir, 'lime')

    # Khởi tạo LIMEExplainer với text mode
    lime_exp = LIMEExplainer(mode="text", class_names=predictor.class_names,  remove_stopwords=True, min_word_length=2)

    # Explain + plot từng sample
    print(f"\n  ── LIME Per-Sample Explanations ──")
    lime_explanations = lime_exp.explain_and_plot_batch(
        instances=texts,
        predict_fn=predictor.predict_proba,
        predict_label_fn=predictor.get_prediction_info,
        num_features=args.lime_num_features,
        num_samples=args.lime_num_samples,
        output_dir=lime_dir,
    )
    print(f"  ✓ LIME per-sample plots saved ({len(lime_explanations)} samples)")

    # ══════════════════════════════════════════════════
    # Step 5: So sánh SHAP vs LIME (side-by-side)
    # ══════════════════════════════════════════════════
    logger.info("Step 5: Comparing SHAP (global) vs LIME (local)...")
    compare_dir = os.path.join(args.output_dir, 'comparison')
    os.makedirs(compare_dir, exist_ok=True)

    # So sánh trên vài sample đầu
    n_compare = min(5, len(texts))
    fig, axes = plt.subplots(n_compare, 2, figsize=(16, 4 * n_compare))
    if n_compare == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_compare):
        # SHAP (per-sample view từ global explanation)
        sv = shap_values[i, :, 'Depression']
        shap_data = [(str(w), float(v)) for w, v in zip(sv.data, sv.values) if str(w).strip()]
        shap_data.sort(key=lambda x: abs(x[1]), reverse=True)
        shap_top = shap_data[:10]

        if shap_top:
            words_s, vals_s = zip(*shap_top)
            colors_s = ['#D85A30' if v > 0 else '#3B8BD4' for v in vals_s]
            axes[i, 0].barh(range(len(words_s)), vals_s, color=colors_s)
            axes[i, 0].set_yticks(range(len(words_s)))
            axes[i, 0].set_yticklabels(words_s)
            axes[i, 0].invert_yaxis()
        axes[i, 0].set_title(f'SHAP — Sample {i+1}', fontsize=10)
        axes[i, 0].set_xlabel('SHAP value')

        # LIME (per-sample)
        lime_feats = lime_explanations[i].as_list(label=1)[:10]
        if lime_feats:
            words_l, vals_l = zip(*lime_feats)
            colors_l = ['#D85A30' if v > 0 else '#3B8BD4' for v in vals_l]
            axes[i, 1].barh(range(len(words_l)), vals_l, color=colors_l)
            axes[i, 1].set_yticks(range(len(words_l)))
            axes[i, 1].set_yticklabels(words_l)
            axes[i, 1].invert_yaxis()
        axes[i, 1].set_title(f'LIME — Sample {i+1}', fontsize=10)
        axes[i, 1].set_xlabel('LIME weight')

    plt.suptitle('So sánh SHAP vs LIME — Top features (Depression)', fontsize=13)
    plt.tight_layout()
    path = os.path.join(compare_dir, 'shap_vs_lime_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Comparison plot saved: {path}")

    # ── Summary ──
    logger.info("Explain pipeline completed!")
    print(f"\n{'='*60}")
    print(f"Tất cả kết quả đã lưu tại: {args.output_dir}")
    print(f"  ├── shap/       — SHAP global importance + waterfall")
    print(f"  │                 (insight cho TOÀN BỘ {len(texts)} samples)")
    print(f"  ├── lime/       — LIME per-sample explanations")
    print(f"  │                 ({len(lime_explanations)} bar charts riêng lẻ)")
    print(f"  └── comparison/ — SHAP vs LIME side-by-side")
    print(f"{'='*60}")


if __name__ == '__main__':
    args = parse_args()
    run_explain_pipeline(args)