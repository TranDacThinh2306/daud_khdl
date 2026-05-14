import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from collections import Counter
from src.utils.logger import setup_logger

logger = setup_logger("depression.features.prototypematcher")

class RAGPrototypeMatcher:
    """
    FAISS-based prototype matcher (in-memory, không Redis).
    Tương thích với interface của PrototypeMatcher gốc.
    """

    def __init__(self, df=None, text_column='post_text', severity_column='weighted_severity'):
        """
        Args:
            df: pandas DataFrame chứa dữ liệu prototypes.
            text_column: tên cột chứa văn bản.
            severity_column: tên cột chứa mức độ trầm cảm.
        """
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_column = text_column
        self.severity_column = severity_column

        self._index = None        
        self._prototypes = []       
        self.database = None       

        if df is not None:
            self.fit(df)

    def _encode(self, texts):
        """Encode và L2-normalize (inner product = cosine)."""
        vecs = self.encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return vecs.astype(np.float32)

    def fit(self, df, force_rebuild=False):
        """
        Xây dựng FAISS index từ dataframe.
        Mỗi dòng trở thành một prototype với các trường:
          - text, severity, naive_score, weighted_score (nếu có)
        """
        if not force_rebuild and self._index is not None:
            print("Index đã tồn tại. Dùng force_rebuild=True để tạo lại.")
            return

        print(f"Đang xây dựng index cho {len(df)} prototypes...")
        self.database = df.reset_index(drop=True)  # đảm bảo index liên tục

        texts = self.database[self.text_column].tolist()
        embeddings = self._encode(texts)

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

        self._prototypes = []
        for idx, row in self.database.iterrows():
            proto = {
                'text': row[self.text_column],
                'severity': row[self.severity_column],
                'naive_score': row.get('naive_phq9', None),
                'weighted_score': row.get('weighted_phq9_norm', None),
            }
            self._prototypes.append(proto)

        print(f"Index sẵn sàng: {self._index.ntotal} vectors, dim={embeddings.shape[1]}")

    def find_prototypes(self, query_text, k=5, return_similarities=True):
        """
        Tìm k prototypes gần nhất với query_text.

        Returns:
            list[dict]: mỗi dict chứa rank, similarity, text, severity,
                        naive_score, weighted_score
        """
        if self._index is None:
            raise RuntimeError("Chưa có index. Hãy gọi fit() trước.")

        query_vec = self._encode([query_text])
        scores, indices = self._index.search(query_vec, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:
                continue
            proto = self._prototypes[idx]
            results.append({
                'rank': rank + 1,
                'similarity': float(score),
                'text': proto['text'],
                'severity': proto['severity'],
                'naive_score': proto['naive_score'],
                'weighted_score': proto['weighted_score']
            })
        return results

    def get_prototype_summary(self, query_text, k=3):
        """
        Trả về summary thống kê của top-k prototypes.

        Returns:
            dict: prototypes, severity_distribution, avg_similarity, most_common_severity
        """
        prototypes = self.find_prototypes(query_text, k=k)

        if not prototypes:
            return {
                'prototypes': [],
                'severity_distribution': {},
                'avg_similarity': 0.0,
                'most_common_severity': None
            }

        severity_counts = Counter(p['severity'] for p in prototypes)
        avg_sim = np.mean([p['similarity'] for p in prototypes])

        return {
            'prototypes': prototypes,
            'severity_distribution': dict(severity_counts),
            'avg_similarity': avg_sim,
            'most_common_severity': severity_counts.most_common(1)[0][0] if severity_counts else None
        }

    def retrieve(self, query_text, top_k=3, severity_filter=None, score_threshold=0.0):
        """Giống find_prototypes nhưng có thêm filter và threshold."""
        results = self.find_prototypes(query_text, k=top_k*2 if severity_filter else top_k)
        filtered = []
        for r in results:
            if r['similarity'] < score_threshold:
                continue
            if severity_filter and r['severity'] != severity_filter:
                continue
            filtered.append(r)
            if len(filtered) == top_k:
                break
        return filtered

    def stats(self):
        """Thông tin về index."""
        if self._index is None:
            return {"total_vectors": 0}
        return {
            "total_vectors": self._index.ntotal,
            "embed_dim": self._index.d,
            "index_type": type(self._index).__name__
        }

    def flush(self):
        """Xoá index trong bộ nhớ."""
        self._index = None
        self._prototypes = []
        self.database = None
        print("Đã flush index.")
