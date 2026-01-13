# MolGen: Flow Matching Molecular Generative Model

A flow matching-based molecular generative model using DiT-style transformers on 2D molecular graphs.

## Quick Start

```bash
# Setup
uv venv && source .venv/bin/activate && uv sync

# Preprocess dataset (one-time, ~3 min for ZINC)
uv run python src/scripts/preprocess_dataset.py zinc --kekulize

# Train
uv run python src/scripts/train_flow.py configs/training/zinc_dit.yaml
```

## Preprocessing

Before training, preprocess the dataset to convert SMILES to tensor format:

```bash
# QM9 (small molecules, 9 atoms max)
uv run python src/scripts/preprocess_dataset.py qm9

# ZINC (drug-like molecules, 38 atoms max, requires kekulization)
uv run python src/scripts/preprocess_dataset.py zinc --kekulize
```

The `--kekulize` flag converts aromatic bonds to alternating single/double bonds (required for ZINC).

Preprocessed files are saved to `data/preprocessed/{dataset}/`:
- `train.pt`, `valid.pt`, `test.pt` - Tensor data
- `train_canonical.json` - Canonical SMILES for novelty computation
- `size_distribution.json` - Molecule size distribution
- `graph_converter.json` - Atom/bond type mappings

## Training

### GraphDiT (Recommended)
```bash
uv run python src/scripts/train_flow.py configs/training/qm9_flow.yaml
uv run python src/scripts/train_flow.py configs/training/zinc_dit.yaml
```

### PairFormer
```bash
uv run python src/scripts/train_flow.py configs/training/qm9_pairformer.yaml
uv run python src/scripts/train_flow.py configs/training/zinc_pairformer.yaml
```

### Multi-GPU Training
```bash
# Run in parallel on different GPUs
CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train_flow.py configs/training/zinc_dit.yaml &
CUDA_VISIBLE_DEVICES=1 uv run python src/scripts/train_flow.py configs/training/zinc_pairformer.yaml &
```

## Project Structure

| Folder | Purpose |
|--------|---------|
| `src/` | All Python source code |
| `src/scripts/` | Entry point scripts (train, generate, evaluate) |
| `src/models/` | Model architectures (DiT, PairFormer, flow matching) |
| `src/data/` | Dataset loaders and graph conversion |
| `configs/` | YAML configuration files |
| `data/` | Experiment outputs and preprocessed data (gitignored) |

## Key Files

- `src/scripts/train_flow.py` - Main training script
- `src/scripts/preprocess_dataset.py` - Dataset preprocessing
- `src/models/dit.py` - GraphDiT transformer architecture
- `src/models/pairformer.py` - PairFormer architecture
- `src/data/graph.py` - Molecule-to-tensor conversion