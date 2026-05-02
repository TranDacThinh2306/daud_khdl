"""
train.py - Huấn luyện mô hình transformer
=============================================
Model training for transformers (e.g., distilbert) with PyTorch.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train sequence classification models (e.g., DistilBERT) using PyTorch."""

    def __init__(
        self, 
        model_name: str = 'distilbert-base-uncased',
        num_labels: int = 2,
        device: Optional[str] = None,
        lr: float = 2e-5,
        epochs: int = 100,
        patience: int = 10,
        save_path: str = 'models_saved/experiments'
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.save_path = save_path
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Init model & tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Optimizer, Loss, Scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Tracking metrics
        self.best_acc = 0.0
        self.train_losses = []
        self.test_accs = []
        self.lrs = []

    def train_epoch(self, train_loader: Any, epoch: int) -> float:
        """Huấn luyện 1 epoch."""
        self.model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}', leave=False)
        
        for batch in loop:
            input_ids = batch['input_ids'].to(self.device)
            attn_mask = batch['attention_mask'].to(self.device)
            labels    = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
            loss    = self.criterion(outputs.logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            loop.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def evaluate(self, val_loader: Any) -> Tuple[float, List, List]:
        """Đánh giá mô hình."""
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attn_mask = batch['attention_mask'].to(self.device)
                outputs   = self.model(input_ids=input_ids, attention_mask=attn_mask)
                preds     = outputs.logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['label'].numpy())

        acc = accuracy_score(all_labels, all_preds)
        return acc, all_preds, all_labels

    def train(self, train_loader: Any, val_loader: Any) -> float:
        """Vòng lặp huấn luyện chính với Early Stopping."""
        logger.info(f"Start training {self.model_name}...")
        patience_count = 0
        
        for epoch in range(self.epochs):
            # Train
            avg_loss = self.train_epoch(train_loader, epoch)
            self.scheduler.step(avg_loss)
            
            # Eval
            acc, _, _ = self.evaluate(val_loader)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.train_losses.append(avg_loss)
            self.test_accs.append(acc)
            self.lrs.append(current_lr)
            
            log_msg = f'Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | LR: {current_lr:.2e}'
            
            # Early stopping & save best model
            if acc > self.best_acc:
                self.best_acc = acc
                patience_count = 0
                os.makedirs(self.save_path, exist_ok=True)
                self.model.save_pretrained(self.save_path)
                self.tokenizer.save_pretrained(self.save_path)
                print(log_msg + '  ✓ Saved (best Acc so far)')
            else:
                patience_count += 1
                print(log_msg + f'  (no improve {patience_count}/{self.patience})')
                if patience_count >= self.patience:
                    print(f'\nEarly stopping tại epoch {epoch+1}!')
                    break

            # Lưu model mỗi 5 epoch để backup
            if (epoch + 1) % 5 == 0:
                backup_path = Path(self.save_path).joinpath(f'epoch_{epoch+1}')
                self.model.save_pretrained(backup_path)
                self.tokenizer.save_pretrained(backup_path)
                print(f'  ✓ Saved backup epoch {epoch+1}')
            
                    
        print(f'\nBest Accuracy: {self.best_acc:.4f} — model đã lưu tại "{self.save_path}"')
        return self.best_acc

    def load_best_model(self):
        """Tải lại mô hình tốt nhất từ thư mục save_path."""
        self.model = AutoModelForSequenceClassification.from_pretrained(self.save_path).to(self.device)
        self.model.eval()
        return self.model

    def evaluate_best_model(self, test_loader: Any):
        """Đánh giá chi tiết mô hình tốt nhất."""
        self.load_best_model()
        acc, all_preds, all_labels = self.evaluate(test_loader)
        
        report = classification_report(
            all_labels, all_preds, target_names=['Non-depression', 'Depression']
        )
        print('\n--- Classification Report (best model) ---')
        print(report)
        return report
