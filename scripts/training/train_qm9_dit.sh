#!/bin/bash
uv run python src/scripts/train_flow.py configs/training/qm9_dit.yaml "$@"
