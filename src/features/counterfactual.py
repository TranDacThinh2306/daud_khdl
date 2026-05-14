import re
import random
import numpy as np
import torch
from transformers import pipeline

from src.utils.logger import setup_logger

logger = setup_logger("depression_alert.features.counterfactual")

severity_mapping = {
    'Minimal': 0,
    'Mild': 1,
    'Moderate': 2,
    'Moderately Severe': 3,
    'Severe': 4
}
# ─────────────────────────────────────────────
# STOPWORD DEFINITIONS
# ─────────────────────────────────────────────
PRONOUNS = {
    "i","me","my","myself","we","our","ours","ourselves",
    "you","your","yours","yourself","yourselves",
    "he","him","his","himself","she","her","hers","herself",
    "it","its","itself","they","them","their","theirs","themselves",
    "who","whom","whose","which","what","this","that","these","those",
    "anyone"
}

PREPOSITIONS = {
    "in","on","at","by","for","with","about","against","between",
    "into","through","during","before","after","above","below",
    "to","from","up","down","of","off","over","under","again",
    "further","then","once","out","as","per","via","near","among",
    "within","without","upon","along","behind","beside","beyond",
    "despite","except","inside","outside","since","toward","until",
    "versus","whereas","around","across","throughout","underneath",
    "else"
}

PUNCTUATION = ["?",".","!"]

ARTICLES = {"a", "an", "the"}

STOPWORDS = PRONOUNS | PREPOSITIONS | ARTICLES


def remove_stopwords(text: str) -> tuple[str, list[str]]:
    """
    Loại bỏ đại từ, giới từ, mạo từ khỏi văn bản.
    Trả về (văn bản đã lọc, danh sách từ đã xóa).
    """
    tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
    kept, removed = [], []
    for token in tokens:
        if token.lower() in STOPWORDS:
            removed.append(token)
        else:
            kept.append(token)
    return " ".join(kept), removed


def is_stopword(token: str) -> bool:
    """Kiểm tra một token (có thể có prefix Ġ của BPE) có phải stopword không."""
    clean = token.strip("Ġ▁").lower()
    clean = re.sub(r"[^a-z]", "", clean)
    return clean in STOPWORDS


# ─────────────────────────────────────────────
# POSITIVE REPLACEMENTS (mở rộng)
# ─────────────────────────────────────────────
POSITIVE_REPLACEMENTS = {
    "sad"         : "happy",
    "sadness"     : "happiness",
    "hopeless"    : "hopeful",
    "hopelessness": "hopefulness",
    "tired"       : "energized",
    "tiring"      : "energizing",
    "exhausted"   : "refreshed",
    "worthless"   : "valuable",
    "failure"     : "success",
    "depressed"   : "content",
    "depression"  : "contentment",
    "anxious"     : "calm",
    "anxiety"     : "calmness",
    "lonely"      : "connected",
    "alone"       : "supported",
    "isolated"    : "connected",
    "empty"       : "fulfilled",
    "burden"      : "blessing",
    "suffering"   : "healing",
    "suffer"      : "recover",
    "pain"        : "comfort",
    "painful"     : "comfortable",
    "miserable"   : "joyful",
    "hate"        : "appreciate",
    "death"       : "life",
    "dead"        : "alive",
    "die"         : "live",
    "dying"       : "living",
    "kill"        : "save",
    "suicide"     : "recovery",
    "cry"         : "smile",
    "crying"      : "smiling",
    "fear"        : "courage",
    "scared"      : "brave",
    "terrified"   : "confident",
    "nightmare"   : "dream",
    "darkness"    : "light",
    "dark"        : "bright",
    "lost"        : "found",
    "losing"      : "gaining",
    "lossing"     : "gaining",
    "loss"        : "gain",
    "numb"        : "feeling",
    "unmotivated" : "motivated",
    "overwhelmed" : "composed",
    "helpless"    : "empowered",
    "trapped"     : "free",
    "broken"      : "whole",
    "destroyed"   : "rebuilt",
    "ruined"      : "restored",
    "awful"       : "wonderful",
    "terrible"    : "great",
    "horrible"    : "amazing",
    "worst"       : "best",
    "bad"         : "good",
    "worse"       : "better",
    "struggling"  : "thriving",
    "struggle"    : "ease",
    "trouble"     : "peace",
}

POSITIVE_PHRASES = [
    " However, I'm feeling much better now.",
    " But things are improving day by day.",
    " I'm receiving treatment and starting to feel hopeful.",
    " My therapist has been very helpful.",
    " I see light at the end of the tunnel.",
    " I've been making great progress in therapy lately.",
    " My support network has really helped me through this.",
    " I'm starting to find joy in small things again.",
    " Every day I'm getting a little stronger.",
    " I've learned healthy coping strategies that really work.",
]


# ─────────────────────────────────────────────
# COUNTERFACTUAL GENERATOR
# ─────────────────────────────────────────────
class CounterfactualGenerator:
    def __init__(self, model, tokenizer, shap_explainer):
        self.model = model
        self.tokenizer = tokenizer
        self.shap_explainer = shap_explainer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.generator = pipeline(
            "text-generation",
            model="distilgpt2",
            max_new_tokens=50
        )

    def find_counterfactual(
        self,
        original_text: str,
        target_severity: int = 0,
        max_attempts: int = 10
    ) -> dict:
        """
        Tìm minimal edits để chuyển prediction sang target_severity.

        Args:
            original_text  : text gốc
            target_severity: 0=Minimal, 1=Mild, 2=Moderate,
                             3=Moderately Severe, 4=Severe
            max_attempts   : số lần thử tối đa
        """
        current_probs   = self.shap_explainer.predict([original_text])[0]
        current_class   = int(np.argmax(current_probs))
        current_severity = list(severity_mapping.keys())[current_class]

        if current_class == target_severity:
            return {
                "success"          : True,
                "message"          : "Already at target severity",
                "original_text"    : original_text,
                "current_severity" : current_severity,
            }

        filtered_text, removed_sw = remove_stopwords(original_text)

        shap_values, _ = self.shap_explainer.explain_text(filtered_text)
        token_shap = shap_values[:, :, current_class].values[0]

        # Lấy token trực tiếp từ SHAP data thay vì re-tokenize
        # để tránh mismatch giữa SHAP masker và tokenizer
        shap_data = shap_values.data
        if isinstance(shap_data, tuple):
            shap_tokens = [str(t) for t in shap_data[0]]
        elif hasattr(shap_data, '__iter__'):
            shap_tokens = [str(t) for t in shap_data]
        else:
            shap_tokens = self.tokenizer.tokenize(filtered_text)

        # Đảm bảo chiều dài token khớp với SHAP values
        min_len = min(len(shap_tokens), len(token_shap))
        shap_tokens = shap_tokens[:min_len]
        token_shap = token_shap[:min_len]

        # Tìm các từ đóng góp tích cực cho predicted class (đẩy về hướng hiện tại)
        negative_words = [
            (token.strip(), float(val))
            for token, val in zip(shap_tokens, token_shap)
            if val > 0
            and token.strip()  # bỏ token rỗng
            and not is_stopword(token)
        ]
        negative_words.sort(key=lambda x: x[1], reverse=True)

        # Nếu không tìm được từ SHAP, fallback: tìm từ tiêu cực trực tiếp trong text
        if not negative_words:
            negative_words = self._find_negative_words_in_text(original_text)

        print(f"Target severity  : {list(severity_mapping.keys())[target_severity]}")
        print(f"Top negative words (after stopword filter): "
              f"{[w for w, _ in negative_words[:20]]}")

        for attempt in range(max_attempts):
            if attempt < 3 and negative_words:
                # Thay thế từ tiêu cực
                modified_text = self._replace_negative_words(
                    original_text, negative_words[: attempt + 2]
                )
            elif attempt < 6 and negative_words:
                # Thay thế từ tiêu cực + thêm câu tích cực
                modified_text = self._replace_negative_words(
                    original_text, negative_words[: attempt + 1]
                )
                modified_text = self._add_positive_phrases(modified_text, count=1)
            else:
                # Thay thế tất cả từ tiêu cực + thêm nhiều câu tích cực
                modified_text = self._replace_negative_words(
                    original_text, negative_words
                )
                modified_text = self._add_positive_phrases(
                    modified_text, count=min(attempt - 4, 3)
                )

            new_probs  = self.shap_explainer.predict([modified_text])[0]
            new_class  = int(np.argmax(new_probs))

            if new_class == target_severity:
                return {
                    "success"          : True,
                    "original_text"    : original_text,
                    "filtered_text"    : filtered_text,
                    "removed_stopwords": removed_sw,
                    "modified_text"    : modified_text,
                    "original_severity": current_severity,
                    "new_severity"     : list(severity_mapping.keys())[new_class],
                    "original_prob"    : float(current_probs[current_class]),
                    "new_prob"         : float(new_probs[new_class]),
                    "changed_words"    : [w for w, _ in negative_words[: attempt + 1]],
                    "attempts"         : attempt + 1,
                }

        return {
            "success"          : False,
            "message"          : f"Could not find counterfactual after {max_attempts} attempts",
            "original_text"    : original_text,
            "current_severity" : current_severity,
        }

    def _find_negative_words_in_text(self, text: str) -> list:
        """
        Fallback: tìm trực tiếp các từ tiêu cực có trong text
        bằng cách so khớp với bảng POSITIVE_REPLACEMENTS.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        found = []
        for word in words:
            if word in POSITIVE_REPLACEMENTS and not is_stopword(word):
                found.append((word, 1.0))  # dummy weight
        # Loại trùng lặp, giữ thứ tự
        seen = set()
        unique = []
        for w, v in found:
            if w not in seen:
                seen.add(w)
                unique.append((w, v))
        return unique

    def _replace_negative_words(self, text: str, negative_words: list) -> str:
        """Thay thế các từ tiêu cực bằng từ tích cực tương ứng."""
        modified = text
        for word, _ in negative_words:
            clean_word = word.strip("Ġ▁").lower()
            clean_word = re.sub(r"[^a-z]", "", clean_word)
            if clean_word in POSITIVE_REPLACEMENTS:
                modified = re.sub(
                    rf"\b{re.escape(clean_word)}\b",
                    POSITIVE_REPLACEMENTS[clean_word],
                    modified,
                    flags=re.IGNORECASE
                )
        return modified

    def _add_positive_phrases(self, text: str, count: int = 1) -> str:
        """Thêm cụm từ tích cực vào cuối văn bản."""
        phrases = random.sample(POSITIVE_PHRASES, min(count, len(POSITIVE_PHRASES)))
        return text + "".join(phrases)
