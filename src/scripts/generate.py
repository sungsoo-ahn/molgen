"""Generate SMILES using trained model."""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.tokenizer import SMILESTokenizer
from src.models.lstm import SMILESLSTM


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

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = SMILESTokenizer()
    tokenizer.load(exp_dir / "tokenizer.json")
    print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")

    # Load model config from experiment
    exp_config_path = exp_dir / "config.yaml"
    with open(exp_config_path, "r") as f:
        exp_config = yaml.safe_load(f)

    model_config = exp_config["model"]

    # Create and load model
    model = SMILESLSTM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=model_config.get("embed_dim", 128),
        hidden_dim=model_config.get("hidden_dim", 256),
        num_layers=model_config.get("num_layers", 2),
        dropout=model_config.get("dropout", 0.2),
        pad_idx=tokenizer.pad_idx,
    ).to(device)

    # Load checkpoint
    gen_config = config.get("generation", {})
    checkpoint_name = gen_config.get("checkpoint", "best_model.pt")
    checkpoint_path = exp_dir / "checkpoints" / checkpoint_name
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"Loaded model from {checkpoint_path}")

    # Generate at multiple temperatures
    num_samples = gen_config.get("num_samples", 10000)
    max_len = gen_config.get("max_len", 100)
    temperatures = gen_config.get("temperatures", [0.7, 0.9, 1.0, 1.2])

    for temp in temperatures:
        print(f"Generating {num_samples} samples at temperature {temp}...")
        samples = model.generate(
            tokenizer,
            num_samples=num_samples,
            max_len=max_len,
            temperature=temp,
            device=device,
        )

        # Save to file
        output_file = output_dir / f"generated_smiles_t{temp}.txt"
        with open(output_file, "w") as f:
            for smiles in samples:
                f.write(smiles + "\n")

        print(f"Saved {len(samples)} samples to {output_file}")

    print(f"Generation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SMILES with trained model")
    parser.add_argument("config_path", type=str, help="Path to config YAML file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
