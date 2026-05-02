"""
scripts/train.py - Chạy training pipeline cho mô hình DistilBERT
==================================================================
Pipeline: Load data → Preprocess → Tokenize → DataLoader → Train → Evaluate
"""

import sys
import os
import logging
import argparse

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Đảm bảo import từ root project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocess import TextPreprocessor
from src.data.load_dataset import get_dataset
from src.data.dataset import DepressionDataset, DataLoader
from src.models.train import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train DistilBERT Depression Classifier')
    parser.add_argument('--dataset', type=str, default='hugginglearners/reddit-depression-cleaned',
                        help='HuggingFace dataset path')
    parser.add_argument('--model_name', type=str, default='models_saved/experiments',
                        help='Pretrained model name')
    parser.add_argument('--epochs', type=int, default=100, help='Max number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='models_saved/experiments',
                        help='Path to save best model')
    return parser.parse_args()


def run_training_pipeline(args):
    """Chạy toàn bộ pipeline huấn luyện."""

    # ── Step 1: Load dataset từ HuggingFace ──
    logger.info("Step 1: Loading dataset...")
    train_df, test_df = get_dataset(args.dataset)
    print(f"  Raw → Train: {len(train_df)} | Test: {len(test_df)}")

    # ── Step 2: Preprocess text ──
    logger.info("Step 2: Preprocessing text...")
    preprocessor = TextPreprocessor()
    train_df = preprocessor.process_dataframe(
        df=train_df, text_column='clean_text',
        output_column='cleaned_text', emoji_processed=True
    )
    test_df = preprocessor.process_dataframe(
        df=test_df, text_column='clean_text',
        output_column='cleaned_text', emoji_processed=True
    )
    print(f"  After preprocess → Train: {len(train_df)} | Test: {len(test_df)}")

    # ── Step 3: Khởi tạo ModelTrainer (tự load model + tokenizer) ──
    logger.info("Step 3: Initializing ModelTrainer...")
    trainer = ModelTrainer(
        model_name=args.model_name,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        save_path=args.save_path
    )
    print(f"  Device: {trainer.device}")
    print(f"  Model: {args.model_name}")

    # ── Step 4: Tạo Dataset + DataLoader ──
    logger.info("Step 4: Creating datasets and dataloaders...")
    train_dataset = DepressionDataset(
        train_df['cleaned_text'], train_df['is_depression'],
        tokenizer=trainer.tokenizer
    )
    test_dataset = DepressionDataset(
        test_df['cleaned_text'], test_df['is_depression'],
        tokenizer=trainer.tokenizer
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size)
    print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Test:  {len(test_dataset)} samples ({len(test_loader)} batches)")

    # ── Step 5: Train với Early Stopping ──
    logger.info("Step 5: Training model...")
    best_acc = trainer.train(train_loader, test_loader)

    # ── Step 6: Evaluate best model ──
    logger.info("Step 6: Evaluating best model...")
    trainer.evaluate_best_model(test_loader)

    # ── Step 7: Visualize kết quả ──
    logger.info("Step 7: Visualizing results...")
    visualize_results(trainer, test_loader, save_dir=args.save_path)

    logger.info("Pipeline completed!")
    return trainer


def visualize_results(trainer, test_loader, save_dir='models_saved/experiments'):
    """Vẽ biểu đồ tổng hợp kết quả huấn luyện."""
    # Lấy predictions từ best model
    _, all_preds, all_labels = trainer.evaluate(test_loader)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- 1. Loss theo epoch ---
    axes[0, 0].plot(trainer.train_losses, color='#D85A30', linewidth=2)
    axes[0, 0].set_title('Training Loss theo epoch')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(alpha=0.3)

    # --- 2. Accuracy theo epoch ---
    axes[0, 1].plot(trainer.test_accs, color='#3B8BD4', linewidth=2)
    axes[0, 1].axhline(trainer.best_acc, color='gray', linestyle='--', linewidth=1)
    axes[0, 1].text(0, trainer.best_acc + 0.002, f'Best: {trainer.best_acc:.4f}', fontsize=9)
    axes[0, 1].set_title('Accuracy theo epoch')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(alpha=0.3)

    # --- 3. Confusion Matrix ---
    cm   = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Non-depression', 'Depression'])
    disp.plot(ax=axes[1, 0], colorbar=False, cmap='Blues')
    axes[1, 0].set_title('Confusion Matrix (best model)')

    # --- 4. Learning Rate theo epoch ---
    axes[1, 1].plot(trainer.lrs, color='#0F6E56', linewidth=2)
    axes[1, 1].set_title('Learning Rate theo epoch')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle('Kết quả huấn luyện — DistilBERT Depression Detection', fontsize=13)
    plt.tight_layout()

    # Lưu vào thư mục save_path
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, 'training_results.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Đã lưu hình vào {save_file}')


if __name__ == '__main__':
    args = parse_args()
    run_training_pipeline(args)
