"""
app/loader.py - Load model và shap_values một lần duy nhất
"""
import os
import sys
import json
import numpy as np
import torch
from typing import List

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import logging

# Tắt cảnh báo từ thư viện transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

# Đảm bảo import từ root project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer


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