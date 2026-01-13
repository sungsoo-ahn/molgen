"""Preprocess molecular datasets to tensor format for fast loading.

This script converts SMILES datasets to preprocessed tensor files,
eliminating the need to re-parse molecules on each training run.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

import torch
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import load_dataset, split_dataset, download_qm9, download_zinc, load_smiles_from_csv
from src.data.graph import (
    ATOM_TYPES_QM9,
    ATOM_TYPES_ZINC,
    BOND_TYPES,
    MolecularGraphConverter,
    get_size_distribution,
)

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")


def preprocess_dataset(
    smiles_list: List[str],
    converter: MolecularGraphConverter,
    desc: str = "Processing",
    kekulize: bool = False,
) -> Dict[str, torch.Tensor]:
    """Convert SMILES list to batched tensors.

    Args:
        smiles_list: List of SMILES strings
        converter: MolecularGraphConverter instance
        desc: Description for progress bar
        kekulize: If True, convert aromatic bonds to single/double bonds

    Returns:
        Dictionary with tensors:
        - node_features: [N, max_atoms, num_atom_types]
        - adjacency: [N, max_atoms, max_atoms, num_bond_types]
        - mask: [N, max_atoms]
        - valid_indices: [N] - original indices of valid molecules
    """
    all_x = []
    all_adj = []
    all_mask = []
    valid_indices = []

    for i, smiles in enumerate(tqdm(smiles_list, desc=desc)):
        result = converter.smiles_to_tensors(smiles, kekulize=kekulize)
        if result is not None:
            x, adj, mask = result
            all_x.append(x)
            all_adj.append(adj)
            all_mask.append(mask)
            valid_indices.append(i)

    return {
        "node_features": torch.stack(all_x),
        "adjacency": torch.stack(all_adj),
        "mask": torch.stack(all_mask),
        "valid_indices": torch.tensor(valid_indices),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess molecular dataset to tensors")
    parser.add_argument(
        "dataset",
        choices=["qm9", "zinc"],
        help="Dataset to preprocess"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory for raw data files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/preprocessed",
        help="Directory for preprocessed files"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction for training set"
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Fraction for validation set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting"
    )
    parser.add_argument(
        "--kekulize",
        action="store_true",
        help="Kekulize molecules (convert aromatic bonds to single/double)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw SMILES
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "qm9":
        csv_path = download_qm9(data_dir)
        atom_types = ATOM_TYPES_QM9
        max_atoms = 9
    else:  # zinc
        csv_path = download_zinc(data_dir)
        atom_types = ATOM_TYPES_ZINC
        max_atoms = 38

    smiles_list = load_smiles_from_csv(csv_path)
    print(f"Loaded {len(smiles_list)} SMILES")

    # Split dataset
    print("Splitting dataset...")
    train_smiles, valid_smiles, test_smiles = split_dataset(
        smiles_list, args.train_ratio, args.valid_ratio, args.seed
    )
    print(f"Train: {len(train_smiles)}, Valid: {len(valid_smiles)}, Test: {len(test_smiles)}")

    # Define bond types (exclude aromatic if kekulizing)
    if args.kekulize:
        from rdkit.Chem import BondType
        bond_types = [None, BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]
        print("Kekulization enabled: using bond types [none, single, double, triple]")
    else:
        bond_types = BOND_TYPES
        print("Using bond types [none, single, double, triple, aromatic]")

    # Create converter
    converter = MolecularGraphConverter(
        atom_types=atom_types,
        bond_types=bond_types,
        max_atoms=max_atoms,
    )

    # Process each split
    dataset_output_dir = output_dir / args.dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    print("\nPreprocessing train set...")
    train_tensors = preprocess_dataset(train_smiles, converter, "Train", kekulize=args.kekulize)
    train_path = dataset_output_dir / "train.pt"
    torch.save(train_tensors, train_path)
    print(f"Saved {len(train_tensors['valid_indices'])} samples to {train_path}")

    print("\nPreprocessing valid set...")
    valid_tensors = preprocess_dataset(valid_smiles, converter, "Valid", kekulize=args.kekulize)
    valid_path = dataset_output_dir / "valid.pt"
    torch.save(valid_tensors, valid_path)
    print(f"Saved {len(valid_tensors['valid_indices'])} samples to {valid_path}")

    print("\nPreprocessing test set...")
    test_tensors = preprocess_dataset(test_smiles, converter, "Test", kekulize=args.kekulize)
    test_path = dataset_output_dir / "test.pt"
    torch.save(test_tensors, test_path)
    print(f"Saved {len(test_tensors['valid_indices'])} samples to {test_path}")

    # Compute and save canonical training set for novelty
    print("\nBuilding canonical training set...")
    train_canonical: Set[str] = set()
    for smiles in tqdm(train_smiles, desc="Canonicalizing"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            train_canonical.add(Chem.MolToSmiles(mol))

    canonical_path = dataset_output_dir / "train_canonical.json"
    with open(canonical_path, "w") as f:
        json.dump(list(train_canonical), f)
    print(f"Saved {len(train_canonical)} canonical SMILES to {canonical_path}")

    # Compute and save size distribution
    print("\nComputing size distribution...")
    size_dist = get_size_distribution(train_smiles, converter)
    dist_path = dataset_output_dir / "size_distribution.json"
    with open(dist_path, "w") as f:
        json.dump(size_dist, f, indent=2)
    print(f"Saved size distribution to {dist_path}")

    # Save converter config
    converter.save(dataset_output_dir / "graph_converter.json")

    # Save metadata
    metadata = {
        "dataset": args.dataset,
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "kekulize": args.kekulize,
        "num_train": len(train_tensors["valid_indices"]),
        "num_valid": len(valid_tensors["valid_indices"]),
        "num_test": len(test_tensors["valid_indices"]),
        "num_canonical": len(train_canonical),
        "atom_types": atom_types,
        "bond_types": [str(b) if b is not None else "none" for b in bond_types],
        "max_atoms": max_atoms,
    }
    with open(dataset_output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nPreprocessing complete! Files saved to {dataset_output_dir}")
    print("Files created:")
    print(f"  - train.pt ({train_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  - valid.pt ({valid_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  - test.pt ({test_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  - train_canonical.json")
    print(f"  - size_distribution.json")
    print(f"  - graph_converter.json")
    print(f"  - metadata.json")


if __name__ == "__main__":
    main()
