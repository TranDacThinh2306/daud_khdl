from datasets import load_dataset
from sklearn.model_selection import train_test_split

def get_dataset(path: str, train_ratio: float = 0.85, test_ratio: float = 0.15):
    dataset = load_dataset(path)
    train_df, test_df = train_test_split(
        dataset['train'].to_pandas(), test_size=test_ratio, random_state=42, stratify=dataset['train']['is_depression']
    )
    return train_df, test_df

