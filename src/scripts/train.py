"""Training orchestration script for SMILES LSTM model."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Set

import torch
import torch.nn as nn
import wandb
import yaml
from rdkit import Chem
from rdkit import RDLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import SMILESDataset, load_dataset
from src.data.tokenizer import SMILESTokenizer
from src.models.lstm import SMILESLSTM

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")


def compute_generation_metrics(
    generated_smiles: List[str],
    train_set: Set[str],
) -> dict:
    """Compute validity, uniqueness, novelty metrics.

    Args:
        generated_smiles: List of generated SMILES
        train_set: Set of canonical training SMILES for novelty

    Returns:
        Dictionary with metrics
    """
    if not generated_smiles:
        return {"validity": 0, "uniqueness": 0, "novelty": 0}

    # Check validity and canonicalize
    valid_canonical = []
    for smiles in generated_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_canonical.append(Chem.MolToSmiles(mol))

    validity = len(valid_canonical) / len(generated_smiles)

    # Uniqueness
    unique_smiles = set(valid_canonical)
    uniqueness = len(unique_smiles) / len(valid_canonical) if valid_canonical else 0

    # Novelty
    novel_smiles = unique_smiles - train_set
    novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0

    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "num_valid": len(valid_canonical),
        "num_unique": len(unique_smiles),
        "num_novel": len(novel_smiles),
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        optimizer.zero_grad()
        logits, _ = model(input_ids)

        # Reshape for cross entropy: [batch * seq_len, vocab_size]
        loss = criterion(logits.view(-1, model.vocab_size), target_ids.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, model.vocab_size), target_ids.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' required in config")

    # Initialize output directory
    output_dir = Path(config["output_dir"])
    if output_dir.exists() and not overwrite:
        raise ValueError(f"Output directory {output_dir} exists. Use --overwrite to overwrite.")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Copy config to output directory
    shutil.copy(config_path, output_dir / "config.yaml")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    wandb_config = config.get("wandb", {})
    if wandb_config:
        wandb.init(
            project=wandb_config.get("project", "molgen"),
            name=wandb_config.get("name", output_dir.name),
            config=config,
        )

    # Load data
    data_dir = Path(config.get("data_dir", "data/raw"))
    dataset_config = config["dataset"]

    print("Loading datasets...")
    train_smiles = load_dataset(
        dataset_config["name"],
        data_dir,
        split="train",
        train_ratio=dataset_config.get("train_split", 0.8),
        valid_ratio=dataset_config.get("valid_split", 0.1),
    )
    valid_smiles = load_dataset(
        dataset_config["name"],
        data_dir,
        split="valid",
        train_ratio=dataset_config.get("train_split", 0.8),
        valid_ratio=dataset_config.get("valid_split", 0.1),
    )

    print(f"Train size: {len(train_smiles)}, Valid size: {len(valid_smiles)}")

    # Build canonical training set for novelty computation
    print("Building canonical training set for novelty computation...")
    train_canonical: Set[str] = set()
    for smiles in tqdm(train_smiles, desc="Canonicalizing", leave=False):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            train_canonical.add(Chem.MolToSmiles(mol))
    print(f"Canonical training set size: {len(train_canonical)}")

    # Build tokenizer
    print("Building tokenizer...")
    tokenizer = SMILESTokenizer()
    tokenizer.build_vocab(train_smiles)
    tokenizer.save(output_dir / "tokenizer.json")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create datasets
    max_len = dataset_config.get("max_len", 128)
    train_dataset = SMILESDataset(train_smiles, tokenizer, max_len=max_len)
    valid_dataset = SMILESDataset(valid_smiles, tokenizer, max_len=max_len)

    training_config = config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    model_config = config["model"]
    model = SMILESLSTM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=model_config.get("embed_dim", 128),
        hidden_dim=model_config.get("hidden_dim", 256),
        num_layers=model_config.get("num_layers", 2),
        dropout=model_config.get("dropout", 0.2),
        pad_idx=tokenizer.pad_idx,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_config.get("learning_rate", 0.001)
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    # Training loop
    best_valid_loss = float("inf")
    patience = training_config.get("early_stopping_patience", 10)
    patience_counter = 0
    history = {"train_loss": [], "valid_loss": []}

    print("Starting training...")
    for epoch in range(training_config["epochs"]):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip=training_config.get("grad_clip", 1.0),
        )
        valid_loss = validate(model, valid_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}")

        # Log to wandb
        if wandb_config:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }

            # Compute generation metrics every 10 epochs
            eval_interval = training_config.get("eval_interval", 10)
            num_eval_samples = training_config.get("num_eval_samples", 1000)

            if (epoch + 1) % eval_interval == 0:
                print(f"  Generating {num_eval_samples} samples for evaluation...")
                samples = model.generate(
                    tokenizer,
                    num_samples=num_eval_samples,
                    temperature=1.0,
                    device=device,
                )
                metrics = compute_generation_metrics(samples, train_canonical)

                log_dict.update({
                    "validity": metrics["validity"],
                    "uniqueness": metrics["uniqueness"],
                    "novelty": metrics["novelty"],
                    "num_valid": metrics["num_valid"],
                    "num_unique": metrics["num_unique"],
                    "num_novel": metrics["num_novel"],
                })

                print(f"  Metrics: validity={metrics['validity']:.2%}, "
                      f"uniqueness={metrics['uniqueness']:.2%}, "
                      f"novelty={metrics['novelty']:.2%}")

                # Log sample molecules
                log_dict["generated_samples"] = wandb.Table(
                    columns=["smiles"], data=[[s] for s in samples[:20]]
                )

            wandb.log(log_dict)

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "checkpoints" / "best_model.pt")
        else:
            patience_counter += 1

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            },
            output_dir / "checkpoints" / "latest_checkpoint.pt",
        )

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Save training history
    with open(output_dir / "results" / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save loss curves
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["valid_loss"], label="Valid Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Loss Curves")
        plt.savefig(output_dir / "figures" / "loss_curves.png", dpi=150)
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping loss curve plot")

    if wandb_config:
        wandb.finish()

    print(f"Training complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SMILES LSTM model")
    parser.add_argument("config_path", type=str, help="Path to config YAML file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
