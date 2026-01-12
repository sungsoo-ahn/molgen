"""Generate molecules using trained flow matching model."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.graph import (
    MolecularGraphConverter,
    create_mask_from_sizes,
    sample_molecule_sizes,
)
from src.models.dit import GraphDiT
from src.models.flow_matching import FlowMatchingSampler, FlowMatchingScheduler


def generate_molecules(
    model: GraphDiT,
    converter: MolecularGraphConverter,
    size_distribution: Dict[int, float],
    num_samples: int,
    device: torch.device,
    num_steps: int = 100,
    method: str = "heun",
    batch_size: int = 64,
) -> List[str]:
    """Generate molecules using flow matching ODE integration.

    Args:
        model: Trained GraphDiT model
        converter: MolecularGraphConverter for tensor-to-SMILES conversion
        size_distribution: Dictionary mapping molecule size to probability
        num_samples: Total number of samples to generate
        device: Device for computation
        num_steps: ODE integration steps
        method: ODE solver method ('euler', 'heun', 'rk4')
        batch_size: Batch size for generation

    Returns:
        List of SMILES strings (empty string for invalid molecules)
    """
    model.eval()
    scheduler = FlowMatchingScheduler()
    sampler = FlowMatchingSampler(model, scheduler, num_steps=num_steps, method=method)

    generated = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Generating"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)

            # Sample molecule sizes from distribution
            sizes = sample_molecule_sizes(
                size_distribution, current_batch_size, device
            )
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


def main(config_path: str) -> None:
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' required in config")

    output_dir = Path(config["output_dir"])
    exp_dir = Path(config.get("exp_dir", output_dir.parent))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load graph converter
    converter_path = exp_dir / "graph_converter.json"
    converter = MolecularGraphConverter.load(converter_path)
    print(f"Loaded graph converter: {converter.num_atom_types} atom types, {converter.max_atoms} max atoms")

    # Load size distribution
    size_dist_path = exp_dir / "size_distribution.json"
    with open(size_dist_path, "r") as f:
        size_distribution_raw = json.load(f)
    # Convert string keys back to int
    size_distribution = {int(k): v for k, v in size_distribution_raw.items()}
    print(f"Loaded size distribution with {len(size_distribution)} sizes")

    # Load model config from experiment
    exp_config_path = exp_dir / "config.yaml"
    with open(exp_config_path, "r") as f:
        exp_config = yaml.safe_load(f)

    model_config = exp_config.get("model", {})

    # Create model
    model = GraphDiT(
        num_atom_types=converter.num_atom_types,
        num_bond_types=converter.num_bond_types,
        max_atoms=converter.max_atoms,
        hidden_dim=model_config.get("hidden_dim", 256),
        num_layers=model_config.get("num_layers", 8),
        num_heads=model_config.get("num_heads", 8),
        mlp_ratio=model_config.get("mlp_ratio", 4.0),
        dropout=model_config.get("dropout", 0.1),
        t_embed_dim=model_config.get("t_embed_dim"),
    ).to(device)

    # Load checkpoint
    gen_config = config.get("generation", {})
    checkpoint_name = gen_config.get("checkpoint", "best_model.pt")
    checkpoint_path = exp_dir / "checkpoints" / checkpoint_name
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model parameters: {model.count_parameters():,}")

    # Generation settings
    num_samples = gen_config.get("num_samples", 10000)
    batch_size = gen_config.get("batch_size", 64)

    # Sampling settings
    sampling_config = config.get("sampling", exp_config.get("sampling", {}))
    num_steps = sampling_config.get("num_steps", 100)
    method = sampling_config.get("method", "heun")

    # Allow multiple num_steps for comparison (optional)
    steps_list = gen_config.get("steps_list", [num_steps])

    for steps in steps_list:
        print(f"\nGenerating {num_samples} samples with {steps} steps, method={method}...")

        samples = generate_molecules(
            model,
            converter,
            size_distribution,
            num_samples=num_samples,
            device=device,
            num_steps=steps,
            method=method,
            batch_size=batch_size,
        )

        # Save to file
        if len(steps_list) == 1:
            output_file = output_dir / "generated_smiles.txt"
        else:
            output_file = output_dir / f"generated_smiles_steps{steps}.txt"

        with open(output_file, "w") as f:
            for smiles in samples:
                f.write(smiles + "\n")

        # Count valid samples
        num_valid = sum(1 for s in samples if s)
        validity = num_valid / len(samples)
        print(f"Saved {len(samples)} samples to {output_file}")
        print(f"Non-empty (potentially valid): {num_valid}/{len(samples)} ({validity:.1%})")

    print(f"\nGeneration complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate molecules with trained flow matching model"
    )
    parser.add_argument("config_path", type=str, help="Path to config YAML file")
    args = parser.parse_args()

    main(args.config_path)
