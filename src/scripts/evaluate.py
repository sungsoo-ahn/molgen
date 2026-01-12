"""Evaluate generated SMILES using validity, uniqueness, novelty metrics."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Set

import yaml
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import load_dataset

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")


def check_validity(smiles: str) -> bool:
    """Check if SMILES is valid using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def canonicalize(smiles: str) -> str:
    """Convert SMILES to canonical form."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)


def compute_metrics(
    generated_smiles: List[str],
    train_smiles: List[str],
) -> dict:
    """Compute evaluation metrics.

    Metrics:
    - Validity: Fraction of valid SMILES (RDKit parseable)
    - Uniqueness: Fraction of unique among valid
    - Novelty: Fraction of unique valid not in training set

    Args:
        generated_smiles: List of generated SMILES strings
        train_smiles: List of training SMILES for novelty computation

    Returns:
        Dictionary of metrics
    """
    if not generated_smiles:
        return {"error": "No generated SMILES provided"}

    # Check validity
    valid_smiles = []
    for smiles in tqdm(generated_smiles, desc="Checking validity"):
        if check_validity(smiles):
            valid_smiles.append(smiles)

    validity = len(valid_smiles) / len(generated_smiles)

    # Canonicalize valid SMILES for uniqueness/novelty
    canonical_valid = []
    for smiles in tqdm(valid_smiles, desc="Canonicalizing"):
        canon = canonicalize(smiles)
        if canon:
            canonical_valid.append(canon)

    # Uniqueness
    unique_smiles = list(set(canonical_valid))
    uniqueness = len(unique_smiles) / len(canonical_valid) if canonical_valid else 0

    # Novelty (compare against training set)
    train_set: Set[str] = set()
    for smiles in tqdm(train_smiles, desc="Canonicalizing training set"):
        canon = canonicalize(smiles)
        if canon:
            train_set.add(canon)

    novel_smiles = [s for s in unique_smiles if s not in train_set]
    novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0

    return {
        "num_generated": len(generated_smiles),
        "num_valid": len(valid_smiles),
        "num_unique": len(unique_smiles),
        "num_novel": len(novel_smiles),
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "valid_unique_novel": validity * uniqueness * novelty,
    }


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' required in config")

    output_dir = Path(config["output_dir"])
    exp_dir = Path(config.get("exp_dir", output_dir.parent))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment config
    exp_config_path = exp_dir / "config.yaml"
    with open(exp_config_path, "r") as f:
        exp_config = yaml.safe_load(f)

    # Load training data for novelty computation
    data_dir = Path(exp_config.get("data_dir", "data/raw"))
    dataset_config = exp_config["dataset"]

    print("Loading training data for novelty computation...")
    train_smiles = load_dataset(
        dataset_config["name"],
        data_dir,
        split="train",
        train_ratio=dataset_config.get("train_split", 0.8),
        valid_ratio=dataset_config.get("valid_split", 0.1),
    )

    # Load generated SMILES
    eval_config = config.get("evaluation", {})
    generated_file = eval_config.get("generated_file", "generated_smiles_t1.0.txt")
    generated_path = exp_dir / generated_file

    if not generated_path.exists():
        raise FileNotFoundError(f"Generated SMILES file not found: {generated_path}")

    print(f"Loading generated SMILES from {generated_path}...")
    with open(generated_path, "r") as f:
        generated_smiles = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(generated_smiles)} generated SMILES")

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(generated_smiles, train_smiles)

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Generated: {metrics['num_generated']}")
    print(f"Valid: {metrics['num_valid']} ({metrics['validity']:.2%})")
    print(f"Unique: {metrics['num_unique']} ({metrics['uniqueness']:.2%})")
    print(f"Novel: {metrics['num_novel']} ({metrics['novelty']:.2%})")
    print(f"Valid * Unique * Novel: {metrics['valid_unique_novel']:.4f}")

    # Save results
    output_file = output_dir / "evaluation_metrics.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated SMILES")
    parser.add_argument("config_path", type=str, help="Path to config YAML file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
