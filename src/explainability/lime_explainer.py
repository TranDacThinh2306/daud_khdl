"""
lime_explainer.py - LIME wrapper (dùng cho từng prediction)
=============================================================
LIME-based explanation for individual predictions.
Supports both tabular (numpy) and text (string) inputs.
"""

import logging
import os
from typing import Any, Callable, List, Optional, Set

import numpy as np
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import lime.lime_text

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Stopword Filter
# ═══════════════════════════════════════════════════════════════════════════

class StopwordFilter:
    """Lọc các từ phổ biến không mang ý nghĩa quyết định trong LIME/SHAP."""
    
    # Stopwords tiếng Anh cơ bản
    ENGLISH_STOPWORDS = {
        'a', 'an', 'and', 'the', 'of', 'to', 'in', 'for', 'on', 'with',
        'at', 'by', 'from', 'up', 'down', 'off', 'over', 'under', 'again',
        'further', 'then', 'now', 'so', 'too', 'also',
    }
    
    # Từ đặc biệt trong ngữ cảnh depression detection
    CUSTOM_STOPWORDS = {
        # Đại từ xưng hô
        'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their',
        'mine', 'yours', 'hers', 'ours', 'theirs',
        
        # Giới từ
        'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'up', 'down', 'off', 'over', 'under', 'again', 'further',
        
        # Liên từ phổ biến
        'and', 'but', 'so', 'or', 'for', 'nor', 'yet',
        'that', 'this', 'these', 'those',
        'a', 'an', 'the',
        
        # Trạng từ chỉ mức độ phổ biến
        'than', 'then', 'now',
        'very', 'just', 'don', 'does', 'did', 'doing',
        'can', 'will', 'would', 'could', 'should',
        'be', 'been', 'being',
        'has', 'have', 'having',
        'was', 'were',
        
        # Từ phổ biến khác
        'yes', 'so', 'too', 'also',
        'here', 'there', 'where',
        'which', 'what', 'who', 'whom',
        
        # Từ viết tắt phổ biến
        'wan', 'ca', 'na', 'gon', 'wanna', 'gonna',
    }
    
    # Từ cần GIỮ LẠI dù là stopword (ví dụ: 'not' rất quan trọng)
    KEEP_WORDS = {
        'not', 'no', 'never', 'nothing', 'none',      # Phủ định rất quan trọng
        "can't", 'cannot', "won't", "wouldn't",       # Phủ định rút gọn
        "don't", "doesn't", "didn't",                 # Phủ định quá khứ
        'but',                                         # Từ nối thể hiện sự đối lập
        'because',                                     # Từ chỉ nguyên nhân
    }
    
    def __init__(self, additional_stopwords: Optional[Set[str]] = None):
        # Hợp nhất stopwords
        self.stopwords = self.ENGLISH_STOPWORDS.union(self.CUSTOM_STOPWORDS)
        
        # Loại bỏ các từ cần giữ lại
        self.stopwords = self.stopwords - self.KEEP_WORDS
        
        # Thêm stopwords tùy chỉnh nếu có
        if additional_stopwords:
            self.stopwords = self.stopwords.union(additional_stopwords)
    
    def should_keep(self, word: str, min_word_length: int = 2) -> bool:
        """Kiểm tra có nên giữ từ này không."""
        word_lower = word.lower()
        return (word_lower not in self.stopwords and 
                len(word_lower) >= min_word_length)
    
    def filter_lime_explanation(self, word_weight_pairs: List[tuple]) -> List[tuple]:
        """Lọc các từ không quan trọng khỏi LIME explanation."""
        filtered = [
            (word, weight) for word, weight in word_weight_pairs
            if self.should_keep(word)
        ]
        return filtered
    
    def filter_and_limit(self, word_weight_pairs: List[tuple], num_features: int = 10) -> List[tuple]:
        """Lọc stopwords và giới hạn số lượng features."""
        filtered = self.filter_lime_explanation(word_weight_pairs)
        return filtered[:num_features]


# ═══════════════════════════════════════════════════════════════════════════
# LIME Explainer
# ═══════════════════════════════════════════════════════════════════════════

class LIMEExplainer:
    """LIME wrapper for explaining individual depression predictions (tabular + text)."""

    def __init__(
        self,
        mode: str = "text",
        training_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        remove_stopwords: bool = True,
        additional_stopwords: Optional[Set[str]] = None,
        min_word_length: int = 2,
    ):
        """
        Args:
            mode: 'text' hoặc 'tabular'
            training_data: Chỉ cần cho mode='tabular'
            feature_names: Tên features (tabular)
            class_names: Tên class labels
            remove_stopwords: Có loại bỏ stopwords không
            additional_stopwords: Stopwords bổ sung
            min_word_length: Độ dài từ tối thiểu để giữ lại
        """
        self.mode = mode
        self.feature_names = feature_names
        self.class_names = class_names or ["Non-depression", "Depression"]
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        
        # Khởi tạo stopword filter nếu cần
        self.stopword_filter = None
        if remove_stopwords and mode == "text":
            self.stopword_filter = StopwordFilter(additional_stopwords)
            logger.info(f"Stopword filter enabled: {len(self.stopword_filter.stopwords)} words filtered")

        if mode == "tabular":
            if training_data is None:
                raise ValueError("training_data is required for tabular mode")
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=feature_names,
                class_names=self.class_names,
                mode="classification",
                discretize_continuous=True,
            )
        elif mode == "text":
            self.explainer = lime.lime_text.LimeTextExplainer(
                class_names=self.class_names,
                split_expression=r'\W+',
                bow=True,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'text' or 'tabular'.")

    def explain_instance(
        self,
        instance,
        predict_fn: Callable,
        num_features: int = 10,
        num_samples: int = 2000,
        labels: tuple = (0, 1),
        filter_stopwords: bool = True,
    ):
        """
        Explain a single prediction using LIME.

        Args:
            instance: Single text (str) hoặc data point (np.ndarray)
            predict_fn: Model's predict_proba function
            num_features: Number of top features to show
            num_samples: Number of perturbation samples for LIME
            labels: Class labels to explain
            filter_stopwords: Áp dụng filter stopwords cho kết quả này

        Returns:
            LIME Explanation object (with filtered features if requested)
        """
        if self.mode == "text":
            explanation = self.explainer.explain_instance(
                instance,
                predict_fn,
                num_features=num_features * 2 if filter_stopwords else num_features,  # Lấy nhiều hơn để sau filter
                num_samples=num_samples,
                labels=labels,
            )
            
            # Lọc stopwords nếu cần
            if filter_stopwords and self.remove_stopwords and self.stopword_filter:
                original_features = explanation.as_list(label=labels[1] if len(labels) > 1 else 1)
                filtered_features = self.stopword_filter.filter_and_limit(
                    original_features, num_features
                )
                # Ghi đè features trong explanation
                explanation._exp_map = dict(filtered_features)
                
                # Log thông tin filter
                filtered_count = len(original_features) - len(filtered_features)
                if filtered_count > 0:
                    logger.debug(f"Filtered out {filtered_count} stopwords from LIME explanation")
        else:
            explanation = self.explainer.explain_instance(
                instance,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
            )
        return explanation

    def explain_batch(
        self,
        instances,
        predict_fn: Callable,
        num_features: int = 10,
        num_samples: int = 2000,
        filter_stopwords: bool = True,
    ) -> list:
        """Explain multiple predictions."""
        explanations = []
        for i, instance in enumerate(instances):
            logger.info(f"LIME explaining instance {i + 1}/{len(instances)}")
            exp = self.explain_instance(
                instance, predict_fn, num_features, num_samples, 
                filter_stopwords=filter_stopwords
            )
            explanations.append(exp)
        return explanations

    def get_top_features(self, explanation, label: int = 1, filter_stopwords: bool = True) -> dict:
        """Extract top features from a LIME explanation."""
        features = explanation.as_list(label=label)
        if filter_stopwords and self.stopword_filter:
            features = self.stopword_filter.filter_and_limit(features, len(features))
        return dict(features)

    def get_top_features_text(self, explanation, label: int = 1, num: int = 5) -> str:
        """Get top features as formatted string."""
        features = self.get_top_features(explanation, label=label)
        top_items = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:num]
        lines = [f"{word:20s} → {weight:+.4f}" for word, weight in top_items]
        return "\n".join(lines)

    # ── Visualization helpers ──

    def plot_explanation(
        self,
        explanation,
        label: int = 1,
        output_dir: str = "reports/figures/lime",
        filename: str = "lime_explanation.png",
        title: Optional[str] = None,
        filter_stopwords: bool = True,
    ) -> str:
        """Save LIME explanation figure (with filtered features)."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Lấy features đã filter để plot
        if filter_stopwords and self.stopword_filter:
            filtered_features = self.stopword_filter.filter_and_limit(
                explanation.as_list(label=label), num_features=10
            )
            # Tạo figure custom để hiển thị features đã filter
            if filtered_features:
                fig, ax = plt.subplots(figsize=(10, max(4, len(filtered_features) * 0.4)))
                words, weights = zip(*filtered_features)
                colors = ['#D85A30' if w > 0 else '#3B8BD4' for w in weights]
                y_pos = range(len(words))
                ax.barh(y_pos, weights, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(words)
                ax.invert_yaxis()
                ax.set_xlabel('LIME weight')
                ax.set_title(title or f'LIME — Top features')
                fig.tight_layout()
                path = os.path.join(output_dir, filename)
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"LIME explanation saved to {path}")
                return path
        
        # Fallback to original LIME plot
        fig = explanation.as_pyplot_figure(label=label)
        if title:
            fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"LIME explanation saved to {path}")
        return path

    def explain_and_plot_batch(
        self,
        instances,
        predict_fn: Callable,
        predict_label_fn: Optional[Callable] = None,
        num_features: int = 10,
        num_samples: int = 2000,
        output_dir: str = "reports/figures/lime",
        filter_stopwords: bool = True,
    ) -> list:
        """Explain + plot cho nhiều samples."""
        os.makedirs(output_dir, exist_ok=True)
        explanations = []

        for i, instance in enumerate(instances):
            logger.info(f"LIME explaining + plotting sample {i+1}/{len(instances)}")

            exp = self.explain_instance(
                instance, predict_fn, num_features, num_samples, 
                filter_stopwords=filter_stopwords
            )
            explanations.append(exp)

            # Tạo title với prediction info
            title = f"LIME — Sample {i+1}"
            if predict_label_fn:
                pred_info = predict_label_fn(instance)
                title += f" ({pred_info})"
            
            self.plot_explanation(
                exp, label=1, output_dir=output_dir,
                filename=f"lime_sample_{i+1}.png", title=title,
                filter_stopwords=filter_stopwords
            )

            # In top features ra console (đã filter)
            print(f"\n  Sample {i+1}: \"{str(instance)[:80]}...\"")
            top_features = self.get_top_features(exp, label=1, filter_stopwords=filter_stopwords)
            for feat, weight in sorted(top_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                direction = "↑ Depression" if weight > 0 else "↓ Non-depression"
                print(f"      {feat:20s} → {weight:+.4f} ({direction})")

        return explanations

    def add_custom_stopwords(self, stopwords: Set[str]):
        """Thêm stopwords tùy chỉnh sau khi khởi tạo."""
        if self.stopword_filter:
            self.stopword_filter.stopwords.update(stopwords)
            logger.info(f"Added {len(stopwords)} custom stopwords")
        else:
            logger.warning("Stopword filter not enabled. Cannot add custom stopwords.")