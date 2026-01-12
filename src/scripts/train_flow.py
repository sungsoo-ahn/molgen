"""Training script for flow matching molecular generative model."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Set

import torch
import torch.nn as nn
import wandb
import yaml
from rdkit import Chem
from rdkit import RDLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import load_dataset
from src.data.graph import (
    ATOM_TYPES_QM9,
    ATOM_TYPES_ZINC,
    BOND_TYPES,
    MolecularGraphConverter,
    MolecularGraphDataset,
    create_mask_from_sizes,
    get_size_distribution,
    sample_molecule_sizes,
)
from src.models.dit import GraphDiT, create_model_from_config
from src.models.pairformer_flow import PairFormerFlow, create_pairformer_from_config
from src.models.flow_matching import (
    FlowMatchingModule,
    FlowMatchingSampler,
    FlowMatchingScheduler,
)

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")


def compute_generation_metrics(
    generated_smiles: List[str],
    train_set: Set[str],
) -> dict:
    """Compute validity, uniqueness, novelty metrics.

    Args:
        generated_smiles: List of generated SMILES (may include empty strings)
        train_set: Set of canonical training SMILES for novelty

    Returns:
        Dictionary with metrics
    """
    # Filter out empty strings (failed conversions)
    non_empty = [s for s in generated_smiles if s]

    if not non_empty:
        return {
            "validity": 0.0,
            "uniqueness": 0.0,
            "novelty": 0.0,
            "num_valid": 0,
            "num_unique": 0,
            "num_novel": 0,
            "num_generated": len(generated_smiles),
        }

    # Check validity and canonicalize
    valid_canonical = []
    for smiles in non_empty:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_canonical.append(Chem.MolToSmiles(mol))

    validity = len(valid_canonical) / len(generated_smiles)

    # Uniqueness
    unique_smiles = set(valid_canonical)
    uniqueness = len(unique_smiles) / len(valid_canonical) if valid_canonical else 0.0

    # Novelty
    novel_smiles = unique_smiles - train_set
    novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0.0

    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "num_valid": len(valid_canonical),
        "num_unique": len(unique_smiles),
        "num_novel": len(novel_smiles),
        "num_generated": len(generated_smiles),
    }


def train_epoch(
    model: FlowMatchingModule,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict:
    """Train for one epoch.

    Returns:
        Dictionary with average loss components
    """
    model.train()
    total_loss = 0.0
    total_loss_nodes = 0.0
    total_loss_edges = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        x = batch["node_features"].to(device)
        adj = batch["adjacency"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        loss, metrics = model.compute_loss(x, adj, mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += metrics["loss_total"]
        total_loss_nodes += metrics["loss_nodes"]
        total_loss_edges += metrics["loss_edges"]
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "loss_nodes": total_loss_nodes / num_batches,
        "loss_edges": total_loss_edges / num_batches,
    }


def validate(
    model: FlowMatchingModule,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate model.

    Returns:
        Dictionary with average loss components
    """
    model.eval()
    total_loss = 0.0
    total_loss_nodes = 0.0
    total_loss_edges = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            x = batch["node_features"].to(device)
            adj = batch["adjacency"].to(device)
            mask = batch["mask"].to(device)

            loss, metrics = model.compute_loss(x, adj, mask)

            total_loss += metrics["loss_total"]
            total_loss_nodes += metrics["loss_nodes"]
            total_loss_edges += metrics["loss_edges"]
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "loss_nodes": total_loss_nodes / num_batches,
        "loss_edges": total_loss_edges / num_batches,
    }


@torch.no_grad()
def generate_samples(
    model: FlowMatchingModule,
    converter: MolecularGraphConverter,
    size_distribution: dict,
    num_samples: int,
    device: torch.device,
    num_steps: int = 100,
    method: str = "heun",
    batch_size: int = 64,
) -> List[str]:
    """Generate molecules using flow matching.

    Args:
        model: Trained FlowMatchingModule
        converter: MolecularGraphConverter for tensor-to-SMILES conversion
        size_distribution: Dictionary mapping molecule size to probability
        num_samples: Total number of samples to generate
        device: Device for computation
        num_steps: ODE integration steps
        method: ODE solver method
        batch_size: Batch size for generation

    Returns:
        List of SMILES strings (empty string for invalid molecules)
    """
    model.eval()
    generated = []

    sampler = FlowMatchingSampler(
        model.model,
        model.scheduler,
        num_steps=num_steps,
        method=method,
    )

    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Generating", leave=False):
        current_batch_size = min(batch_size, num_samples - i * batch_size)

        # Sample molecule sizes
        sizes = sample_molecule_sizes(size_distribution, current_batch_size, device)
        mask = create_mask_from_sizes(sizes, converter.max_atoms)

        # Generate via ODE integration
        x, adj = sampler.sample(current_batch_size, mask, device)

        # Convert to SMILES
        for j in range(current_batch_size):
            smiles = converter.tensors_to_smiles(
                x[j].cpu(),
                adj[j].cpu(),
                mask[j].cpu(),
            )
            generated.append(smiles if smiles else "")

    return generated


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create cosine learning rate schedule with warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main(config_path: str, overwrite: bool = False) -> None:
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' required in config")

    # Initialize output directory
    output_dir = Path(config["output_dir"])
    if output_dir.exists() and not overwrite:
        raise ValueError(
            f"Output directory {output_dir} exists. Use --overwrite to overwrite."
        )

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
    graph_config = config.get("graph", {})

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

    # Create graph converter
    print("Creating graph converter...")

    # Determine atom types based on dataset
    dataset_name = dataset_config["name"].lower()
    if "atom_types" in graph_config:
        atom_types = graph_config["atom_types"]
    elif dataset_name == "qm9":
        atom_types = ATOM_TYPES_QM9
    else:
        atom_types = ATOM_TYPES_ZINC

    max_atoms = graph_config.get("max_atoms", 9 if dataset_name == "qm9" else 38)

    converter = MolecularGraphConverter(
        atom_types=atom_types,
        bond_types=BOND_TYPES,
        max_atoms=max_atoms,
    )
    converter.save(output_dir / "graph_converter.json")

    print(f"Atom types: {atom_types}")
    print(f"Max atoms: {max_atoms}")

    # Get molecule size distribution
    print("Computing molecule size distribution...")
    size_distribution = get_size_distribution(train_smiles, converter)
    print(f"Size distribution: {size_distribution}")

    # Save size distribution
    with open(output_dir / "size_distribution.json", "w") as f:
        json.dump(size_distribution, f, indent=2)

    # Create datasets
    print("Creating datasets...")
    train_dataset = MolecularGraphDataset(train_smiles, converter)
    valid_dataset = MolecularGraphDataset(valid_smiles, converter)

    print(f"Valid training samples: {len(train_dataset)}")
    print(f"Valid validation samples: {len(valid_dataset)}")

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
    print("Creating model...")
    model_config = config.get("model", {})
    architecture = model_config.get("architecture", "graphdit")

    if architecture == "pairformer":
        # Create PairFormer model
        backbone = PairFormerFlow(
            num_atom_types=converter.num_atom_types,
            num_bond_types=converter.num_bond_types,
            max_atoms=converter.max_atoms,
            hidden_dim_s=model_config.get("hidden_dim_s", model_config.get("hidden_dim", 256)),
            hidden_dim_z=model_config.get("hidden_dim_z", 128),
            num_layers=model_config.get("num_layers", 8),
            num_heads=model_config.get("num_heads", 8),
            mlp_ratio=model_config.get("mlp_ratio", 4.0),
            dropout=model_config.get("dropout", 0.1),
            t_embed_dim=model_config.get("t_embed_dim"),
            use_triangle_mult=model_config.get("use_triangle_mult", True),
            use_triangle_attn=model_config.get("use_triangle_attn", True),
        ).to(device)
        print(f"Architecture: PairFormer (triangle_mult={model_config.get('use_triangle_mult', True)}, "
              f"triangle_attn={model_config.get('use_triangle_attn', True)})")
    else:
        # Create GraphDiT model (default)
        # Extract positional encoding config
        pe_config = model_config.get("positional_encoding", {})
        pe_type = pe_config.get("type", "learnable")

        backbone = GraphDiT(
            num_atom_types=converter.num_atom_types,
            num_bond_types=converter.num_bond_types,
            max_atoms=converter.max_atoms,
            hidden_dim=model_config.get("hidden_dim", 256),
            num_layers=model_config.get("num_layers", 8),
            num_heads=model_config.get("num_heads", 8),
            mlp_ratio=model_config.get("mlp_ratio", 4.0),
            dropout=model_config.get("dropout", 0.1),
            t_embed_dim=model_config.get("t_embed_dim"),
            pe_type=pe_type,
            pe_config=pe_config,
        ).to(device)
        print(f"Architecture: GraphDiT (PE type: {pe_type})")

    # Create flow matching wrapper
    fm_config = config.get("flow_matching", {})
    scheduler = FlowMatchingScheduler(sigma_min=fm_config.get("sigma_min", 0.001))
    model = FlowMatchingModule(backbone, scheduler)

    print(f"Model parameters: {backbone.count_parameters():,}")

    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.get("learning_rate", 0.0001),
        weight_decay=training_config.get("weight_decay", 0.01),
    )

    # Setup learning rate scheduler
    num_epochs = training_config["epochs"]
    warmup_epochs = training_config.get("warmup_epochs", 10)
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Sampling config
    sampling_config = config.get("sampling", {})
    num_steps = sampling_config.get("num_steps", 100)
    method = sampling_config.get("method", "heun")

    # Training loop
    best_valid_loss = float("inf")
    patience = training_config.get("early_stopping_patience", 20)
    patience_counter = 0
    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_loss_nodes": [],
        "train_loss_edges": [],
        "valid_loss_nodes": [],
        "valid_loss_edges": [],
    }

    print("Starting training...")
    for epoch in range(num_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip=training_config.get("grad_clip", 1.0),
        )

        # Step LR scheduler
        lr_scheduler.step()

        valid_metrics = validate(model, valid_loader, device)

        history["train_loss"].append(train_metrics["loss"])
        history["valid_loss"].append(valid_metrics["loss"])
        history["train_loss_nodes"].append(train_metrics["loss_nodes"])
        history["train_loss_edges"].append(train_metrics["loss_edges"])
        history["valid_loss_nodes"].append(valid_metrics["loss_nodes"])
        history["valid_loss_edges"].append(valid_metrics["loss_edges"])

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"valid_loss={valid_metrics['loss']:.4f}, "
            f"lr={current_lr:.2e}"
        )

        # Log to wandb
        if wandb_config:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "valid_loss": valid_metrics["loss"],
                "train_loss_nodes": train_metrics["loss_nodes"],
                "train_loss_edges": train_metrics["loss_edges"],
                "valid_loss_nodes": valid_metrics["loss_nodes"],
                "valid_loss_edges": valid_metrics["loss_edges"],
                "learning_rate": current_lr,
            }

            # Compute generation metrics periodically
            eval_interval = training_config.get("eval_interval", 10)
            num_eval_samples = training_config.get("num_eval_samples", 1000)

            if (epoch + 1) % eval_interval == 0:
                print(f"  Generating {num_eval_samples} samples for evaluation...")
                samples = generate_samples(
                    model,
                    converter,
                    size_distribution,
                    num_samples=num_eval_samples,
                    device=device,
                    num_steps=num_steps,
                    method=method,
                    batch_size=64,
                )
                metrics = compute_generation_metrics(samples, train_canonical)

                log_dict.update(
                    {
                        "validity": metrics["validity"],
                        "uniqueness": metrics["uniqueness"],
                        "novelty": metrics["novelty"],
                        "num_valid": metrics["num_valid"],
                        "num_unique": metrics["num_unique"],
                        "num_novel": metrics["num_novel"],
                    }
                )

                print(
                    f"  Metrics: validity={metrics['validity']:.2%}, "
                    f"uniqueness={metrics['uniqueness']:.2%}, "
                    f"novelty={metrics['novelty']:.2%}"
                )

                # Log sample molecules
                valid_samples = [s for s in samples[:100] if s]
                if valid_samples:
                    log_dict["generated_samples"] = wandb.Table(
                        columns=["smiles"], data=[[s] for s in valid_samples[:20]]
                    )

            wandb.log(log_dict)

        # Save best model
        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            patience_counter = 0
            torch.save(
                model.model.state_dict(),
                output_dir / "checkpoints" / "best_model.pt",
            )
        else:
            patience_counter += 1

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "train_loss": train_metrics["loss"],
                "valid_loss": valid_metrics["loss"],
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

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Total loss
        axes[0].plot(history["train_loss"], label="Train Loss")
        axes[0].plot(history["valid_loss"], label="Valid Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].set_title("Total Loss")

        # Component losses
        axes[1].plot(history["train_loss_nodes"], label="Train Nodes", linestyle="-")
        axes[1].plot(history["train_loss_edges"], label="Train Edges", linestyle="--")
        axes[1].plot(history["valid_loss_nodes"], label="Valid Nodes", linestyle="-")
        axes[1].plot(history["valid_loss_edges"], label="Valid Edges", linestyle="--")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].set_title("Component Losses")

        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "loss_curves.png", dpi=150)
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping loss curve plot")

    if wandb_config:
        wandb.finish()

    print(f"Training complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flow matching molecular model")
    parser.add_argument("config_path", type=str, help="Path to config YAML file")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output"
    )
    args = parser.parse_args()

    main(args.config_path, args.overwrite)
