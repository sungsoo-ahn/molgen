"""PyTorch Dataset classes for QM9 and ZINC SMILES data."""

import random
from pathlib import Path
from typing import List, Tuple

import requests
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.tokenizer import SMILESTokenizer

# Dataset URLs
QM9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
ZINC250K_URL = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"


def download_file(url: str, path: Path) -> None:
    """Download file from URL with progress bar."""
    path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_qm9(data_dir: Path) -> Path:
    """Download QM9 dataset and return path to CSV."""
    data_dir = Path(data_dir)
    csv_path = data_dir / "qm9.csv"

    if not csv_path.exists():
        print(f"Downloading QM9 dataset to {csv_path}...")
        download_file(QM9_URL, csv_path)

    return csv_path


def download_zinc(data_dir: Path) -> Path:
    """Download ZINC250K dataset and return path to CSV."""
    data_dir = Path(data_dir)
    csv_path = data_dir / "zinc250k.csv"

    if not csv_path.exists():
        print(f"Downloading ZINC250K dataset to {csv_path}...")
        download_file(ZINC250K_URL, csv_path)

    return csv_path


def load_smiles_from_csv(csv_path: Path, smiles_col: str = "smiles") -> List[str]:
    """Load SMILES strings from CSV file."""
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Try common column names
    for col in [smiles_col, "SMILES", "smiles", "canonical_smiles", "mol"]:
        if col in df.columns:
            return df[col].dropna().tolist()

    raise ValueError(f"Could not find SMILES column in {csv_path}. Columns: {df.columns.tolist()}")


def split_dataset(
    smiles_list: List[str],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split SMILES list into train/valid/test sets."""
    random.seed(seed)
    shuffled = smiles_list.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)

    return shuffled[:train_end], shuffled[train_end:valid_end], shuffled[valid_end:]


def load_dataset(
    name: str,
    data_dir: Path,
    split: str = "train",
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
    max_samples: int = None,
) -> List[str]:
    """Load SMILES dataset by name.

    Args:
        name: 'qm9' or 'zinc'
        data_dir: Directory to store/load data
        split: 'train', 'valid', or 'test'
        train_ratio: Fraction for training set
        valid_ratio: Fraction for validation set
        seed: Random seed for splitting
        max_samples: If provided, limit total samples before splitting (for quick testing)

    Returns:
        List of SMILES strings
    """
    data_dir = Path(data_dir)

    if name.lower() == "qm9":
        csv_path = download_qm9(data_dir)
    elif name.lower() in ["zinc", "zinc250k"]:
        csv_path = download_zinc(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'qm9' or 'zinc'.")

    smiles_list = load_smiles_from_csv(csv_path)

    # Limit total samples if specified (for quick testing)
    if max_samples is not None and max_samples < len(smiles_list):
        random.seed(seed)
        smiles_list = random.sample(smiles_list, max_samples)

    train_smiles, valid_smiles, test_smiles = split_dataset(
        smiles_list, train_ratio, valid_ratio, seed
    )

    if split == "train":
        return train_smiles
    elif split == "valid":
        return valid_smiles
    elif split == "test":
        return test_smiles
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'valid', or 'test'.")


class SMILESDataset(Dataset):
    """PyTorch Dataset for SMILES sequences.

    Args:
        smiles_list: List of SMILES strings
        tokenizer: SMILESTokenizer instance
        max_len: Maximum sequence length (pad/truncate)
    """

    def __init__(
        self,
        smiles_list: List[str],
        tokenizer: SMILESTokenizer,
        max_len: int = 128,
    ):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> dict:
        smiles = self.smiles_list[idx]
        tokens = self.tokenizer.encode(smiles, add_special=True)

        # Truncate or pad
        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len]
        else:
            tokens = tokens + [self.tokenizer.pad_idx] * (self.max_len - len(tokens))

        # For language modeling: input is tokens[:-1], target is tokens[1:]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)

        return {"input_ids": input_ids, "target_ids": target_ids}
