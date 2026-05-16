"""
app/loader.py - Load model và shap_values một lần duy nhất
"""
import os
import sys
import json
import numpy as np
import torch
import pandas as pd
from typing import List
from collections import Counter

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline as hf_pipeline

import logging

# Tắt cảnh báo từ thư viện transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

# Đảm bảo import từ root project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.features import RAGPrototypeMatcher


# ── Device ──
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── TransformerPredictor (copy từ run_explain.py) ──
class TransformerPredictor:
    def __init__(self, model, tokenizer, device, max_len=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.class_names = ['Non-depression', 'Depression']

    def predict_proba(self, texts: List[str]) -> np.ndarray:
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
        if isinstance(texts, str):
            texts = [texts]
        return self.predict_proba(list(texts))

    def get_prediction_info(self, text: str) -> str:
        probs = self.predict_proba([text])[0]
        pred_label = int(probs.argmax())
        return f"Pred: {self.class_names[pred_label]}, P={probs[pred_label]:.3f}"


def load_model(model_path: str):
    """Load DistilBERT model + tokenizer."""
    device = get_device()
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    predictor = TransformerPredictor(model, tokenizer, device)
    return predictor, tokenizer, device


def load_global_importance(json_path: str) -> dict:
    """Load global_importance.json đã tính sẵn — nhanh hơn tính lại từ pkl."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def preprocess_input(text: str) -> str:
    """
    Tạo dataframe giả 1 dòng → chạy qua TextPreprocessor → lấy cleaned_text.
    Đảm bảo nhất quán với pipeline lúc train model.
    """
    import pandas as pd
    from src.data.preprocess import TextPreprocessor

    preprocessor = TextPreprocessor()
    df_fake = pd.DataFrame({'clean_text': [text]})
    try:
        df_processed = preprocessor.process_dataframe(
            df=df_fake,
            text_column='clean_text',
            output_column='cleaned_text',
            emoji_processed=False,  # không cần extract emoji features lúc inference
            log_path=None           # không cần log lúc inference
        )
        if df_processed.empty or df_processed['cleaned_text'].iloc[0].strip() == '':
            return text  # fallback về text gốc nếu clean xong rỗng
        return df_processed['cleaned_text'].iloc[0]
    except Exception:
        return text  # fallback an toàn nếu có lỗi


def run_lime(predictor: TransformerPredictor, text: str,
             num_features: int = 10, num_samples: int = 1000):
    """Chạy LIME real-time cho 1 câu."""
    lime_exp = LIMEExplainer(
        mode="text",
        class_names=predictor.class_names,
        remove_stopwords=False,
        min_word_length=1
    )
    exp = lime_exp.explain_instance(
        instance=text,
        predict_fn=predictor.predict_proba,
        num_features=num_features,
        num_samples=num_samples
    )
    return exp


def run_shap_local(predictor: TransformerPredictor, tokenizer, text: str,
                   max_evals: int = 300):
    """Chạy SHAP local real-time cho 1 câu."""
    shap_exp = SHAPExplainer(model=predictor, explainer_type="text")
    shap_exp.fit_text(
        predict_fn=predictor,
        tokenizer=tokenizer,
        output_names=predictor.class_names
    )
    sv = shap_exp.explain([text], max_evals=max_evals)
    return sv

# ════════════════════════════════════════════════════
# HƯỚNG 2 — RAG Prototype Matching + MentalBERT
# ════════════════════════════════════════════════════

# Mapping PHQ-9
PHQ_TO_ITEM = {
    "Little-interest-or-pleasure-in-doing": 1,
    "Feeling-down-depressed-or-hopeless": 2,
    "Trouble-falling-or-staying-asleep-or-sleeping-too-much": 3,
    "Feeling-tired-or-having-little-energy": 4,
    "Poor-appetite-or-overeating": 5,
    "Feeling-bad-about-yourself-or-that-you-are-a-failure-or-have-let-yourself-or-your-family-down": 6,
    "Trouble-concentrating-on-things-such-as-reading-the-newspaper-or-watching-television": 7,
    "Moving-or-speaking-so-slowly-that-other-people-could-have-noticed-Or-the-opposite-being-so-fidgety-or-restless-that-you-have-been-moving-around-a-lot-more-than-usual": 8,
    "Thoughts-that-you-would-be-better-off-dead-or-of-hurting-yourself-in-some-way": 9,
}

WEIGHTS_WITH_PHQ = {
    "Little-interest-or-pleasure-in-doing": 0.93,
    "Feeling-down-depressed-or-hopeless": 1.00,
    "Trouble-falling-or-staying-asleep-or-sleeping-too-much": 0.86,
    "Feeling-tired-or-having-little-energy": 0.99,
    "Poor-appetite-or-overeating": 0.86,
    "Feeling-bad-about-yourself-or-that-you-are-a-failure-or-have-let-yourself-or-your-family-down": 0.92,
    "Trouble-concentrating-on-things-such-as-reading-the-newspaper-or-watching-television": 0.87,
    "Moving-or-speaking-so-slowly-that-other-people-could-have-noticed-Or-the-opposite-being-so-fidgety-or-restless-that-you-have-been-moving-around-a-lot-more-than-usual": 0.63,
    "Thoughts-that-you-would-be-better-off-dead-or-of-hurting-yourself-in-some-way": 0.96,
}

SEVERITY_MAPPING = {
    "Minimal": 0,
    "Mild": 1,
    "Moderate": 2,
    "Moderately Severe": 3,
    "Severe": 4,
}

PHQ9_COLS = list(PHQ_TO_ITEM.keys())


def preprocess_phq9_data(
    dataset_name: str = "darssanle/PHQ-9-Initial-Collection",
    split: str = "train",
    verbose: bool = True,
) -> pd.DataFrame:
    """Load và preprocess dataset PHQ-9 từ HuggingFace."""
    from datasets import load_dataset

    if verbose:
        print(f"Loading dataset '{dataset_name}' (split='{split}')...")
    dataset = load_dataset(dataset_name, split=split)
    df = pd.DataFrame(dataset)
    if verbose:
        print(f"  → Total samples: {len(df)}")

    def _calculate_weighted_score(annotations: list) -> dict:
        weighted_sum = 0.0
        naive_count = 0
        for question_text, answer in annotations:
            score = 1 if answer.lower() == "yes" else 0
            weight = WEIGHTS_WITH_PHQ.get(question_text, 1.0)
            weighted_sum += score * weight
            naive_count += score
        normalized_score = (weighted_sum / 9) * 27 if 9 > 0 else 0
        return {
            "raw_weighted_score": weighted_sum,
            "normalized_score": normalized_score,
            "naive_score": naive_count,
        }

    def _classify_severity(score: float) -> str:
        if score <= 4:
            return "Minimal"
        elif score <= 9:
            return "Mild"
        elif score <= 14:
            return "Moderate"
        elif score <= 19:
            return "Moderately Severe"
        else:
            return "Severe"

    phq9_scores = df["annotations"].apply(_calculate_weighted_score)
    df["naive_phq9"]        = phq9_scores.apply(lambda x: x["naive_score"])
    df["weighted_phq9_raw"] = phq9_scores.apply(lambda x: x["raw_weighted_score"])
    df["weighted_phq9_norm"]= phq9_scores.apply(lambda x: x["normalized_score"])
    df["weighted_severity"] = df["weighted_phq9_norm"].apply(_classify_severity)
    df["label"]             = df["weighted_severity"].map(SEVERITY_MAPPING)

    if verbose:
        print(f"✅ Preprocessing complete! Shape: {df.shape}")
    return df


def load_prototype_matcher(df: pd.DataFrame) -> RAGPrototypeMatcher:
    """Khởi tạo RAGPrototypeMatcher và build FAISS index từ DataFrame."""
    matcher = RAGPrototypeMatcher(
        df,
        text_column="post_text",
        severity_column="weighted_severity"
    )
    return matcher


class MentalBertPipeline:
    """Wrapper cho MentalBERT fine-tuned để classify mức độ trầm cảm."""

    LABEL_MAP = {
        "LABEL_0": "Minimal",
        "LABEL_1": "Mild",
        "LABEL_2": "Moderate",
        "LABEL_3": "Moderately Severe",
        "LABEL_4": "Severe",
    }

    def __init__(self, model_path: str):
        self.classifier = hf_pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=0 if torch.cuda.is_available() else -1,
        )

    def predict(self, text: str) -> dict:
        raw = self.classifier(text, truncation=True, max_length=512)[0]
        return {
            "severity": self.LABEL_MAP.get(raw["label"], raw["label"]),
            "confidence": raw["score"],
            "raw_label": raw["label"],
        }


def load_mentalbert(model_path: str) -> MentalBertPipeline:
    """Load MentalBERT fine-tuned từ thư mục local."""
    return MentalBertPipeline(model_path)


def row_to_binary(ann, phq_cols: list) -> list:
    """Chuyển annotations thành vector binary 0/1 theo thứ tự phq_cols."""
    import ast
    if isinstance(ann, str):
        try:
            ann = ast.literal_eval(ann)
        except Exception:
            ann = []
    if not isinstance(ann, list):
        ann = []
    mapping = {symptom: 1 if label.lower() == "yes" else 0 for symptom, label in ann}
    return [mapping.get(col, 0) for col in phq_cols]


def run_prototype_matching(
    df: pd.DataFrame,
    matcher: RAGPrototypeMatcher,
    query_text: str,
    top_k: int = 5,
) -> dict:
    """
    Chạy toàn bộ pipeline RAG Prototype Matching cho 1 câu query.
    Trả về dict gồm:
        - prototypes: list top-k kết quả
        - phq_scores: list (question, score) sắp xếp giảm dần
        - rag_result: dict weighted_phq9 classify
    """
    # Tìm top-k prototype
    prototype_summary = matcher.get_prototype_summary(query_text, k=top_k + 1)
    top_match = prototype_summary["prototypes"][1:]  # bỏ bản thân nếu có

    if not top_match:
        return {"prototypes": [], "phq_scores": [], "rag_result": None}

    # Merge với df để lấy annotations
    top_match_df = pd.DataFrame(top_match)
    df_joined = df.merge(top_match_df, left_on="post_text", right_on="text", how="inner")
    df_joined = df_joined.drop(columns=["text"], errors="ignore")

    binary_data = df_joined["annotations"].apply(
        lambda x: row_to_binary(x, PHQ9_COLS)
    ).tolist()
    df_joined[PHQ9_COLS] = pd.DataFrame(binary_data, columns=PHQ9_COLS, index=df_joined.index)

    # Tính score từng triệu chứng
    length = len(df_joined)
    phq_scores = {}
    for col in PHQ9_COLS:
        phq_scores[col] = (df_joined[col] * df_joined["similarity"]).sum() / length * WEIGHTS_WITH_PHQ[col]
    phq_scores_sorted = sorted(phq_scores.items(), key=lambda x: x[1], reverse=True)

    # Classify mức độ từ weighted score
    total_score = sum(v for _, v in phq_scores_sorted)
    normalized = (total_score / 9) * 27

    def _classify(score):
        if score <= 4:   return "Minimal"
        elif score <= 9: return "Mild"
        elif score <= 14: return "Moderate"
        elif score <= 19: return "Moderately Severe"
        else:            return "Severe"

    rag_result = {
        "weighted_score": total_score,
        "normalized_score": normalized,
        "severity": _classify(normalized),
        "most_common_severity": prototype_summary["most_common_severity"],
        "avg_similarity": prototype_summary["avg_similarity"],
    }

    return {
        "prototypes": top_match,
        "phq_scores": phq_scores_sorted,
        "rag_result": rag_result,
    }