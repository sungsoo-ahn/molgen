#!/bin/bash
uv run python src/scripts/train_flow.py configs/training/zinc_pairmixer.yaml "$@"
