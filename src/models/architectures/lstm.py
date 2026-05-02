"""
lstm.py - LSTM cho sequential text
=====================================
LSTM model for sequential text classification.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_lstm(
    vocab_size: int = 30000,
    embedding_dim: int = 300,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    bidirectional: bool = True,
    pretrained_embeddings=None,
):
    """
    Build an LSTM model for sequential text classification.

    Args:
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        bidirectional: Use bidirectional LSTM
        pretrained_embeddings: Pre-trained embedding matrix

    Returns:
        DepressionLSTM model
    """
    import torch
    import torch.nn as nn

    class DepressionLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            if pretrained_embeddings is not None:
                self.embedding.weight = nn.Parameter(
                    torch.tensor(pretrained_embeddings, dtype=torch.float32)
                )
                self.embedding.weight.requires_grad = False

            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
            direction_factor = 2 if bidirectional else 1
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * direction_factor, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, (hidden, _) = self.lstm(embedded)
            if self.lstm.bidirectional:
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                hidden = hidden[-1]
            out = self.dropout(hidden)
            out = self.fc(out)
            return self.sigmoid(out).squeeze()

    model = DepressionLSTM()
    logger.info(f"Built LSTM model with {sum(p.numel() for p in model.parameters())} parameters")
    return model
