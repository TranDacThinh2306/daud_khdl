"""
bert_model.py - Fine-tune BERT/StructBERT
============================================
BERT-based model for depression detection via fine-tuning.
"""

import logging
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
logger = logging.getLogger(__name__)


MODEL_NAME = 'distilbert-base-uncased'  # Nhẹ hơn BERT gốc, phù hợp Colab free

def build_bert_model(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
    dropout: float = 0.1,
    freeze_base: bool = False,
):
    """
    Build a BERT classifier for depression detection.

    Args:
        model_name: HuggingFace model name (e.g., 'bert-base-uncased')
        num_labels: Number of output labels
        dropout: Dropout rate
        freeze_base: Freeze BERT base layers

    Returns:
        Configured AutoModelForSequenceClassification model
    """
    from transformers import AutoModelForSequenceClassification, AutoConfig

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    
    # Thiết lập dropout tương ứng tùy thuộc vào loại model
    if hasattr(config, "classifier_dropout"):
        config.classifier_dropout = dropout
    elif hasattr(config, "seq_classif_dropout"):
        config.seq_classif_dropout = dropout
    elif hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = dropout

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )

    if freeze_base:
        if hasattr(model, 'base_model'):
            for param in model.base_model.parameters():
                param.requires_grad = False
            logger.info(f"{model_name} base layers frozen")
        else:
            logger.warning(f"Could not find base_model for {model_name} to freeze")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"BERT model: {total_params} total, {trainable} trainable parameters")

    return model

