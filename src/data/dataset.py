from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union, Any, List
import pandas as pd


class Config:
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LEN = 128

# Dataset class
class DepressionDataset(Dataset):
    def __init__(self, 
    texts: Union[pd.Series, List[str]], 
    labels: Union[pd.Series, List[int]], 
    tokenizer: AutoTokenizer, 
    max_len: int = Config.MAX_LEN):
        if type(texts) is pd.Series:
            self.texts = texts.tolist()
        elif type(texts) is list:
            self.texts = texts
        else:
            raise TypeError("texts must be a pandas Series, or a list")
            
        if type(labels) is pd.Series:
            self.labels = labels.tolist()
        elif type(labels) is list:
            self.labels = labels
        else:
            raise TypeError("labels must be a pandas Series, or a list")
            
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

# train_dataset = DepressionDataset(
#     df_train[text_col].tolist(),
#     df_train['label'].tolist(),
#     tokenizer, MAX_LEN
# )
# test_dataset = DepressionDataset(
#     df_test[text_col].tolist(),
#     df_test['label'].tolist(),
#     tokenizer, MAX_LEN
# )

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# print(f'Train: {len(train_dataset)} samples | Test: {len(test_dataset)} samples')