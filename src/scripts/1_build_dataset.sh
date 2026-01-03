#!/usr/bin/env bash
set -e
python -m src.data.build_dataset \
  --train_in data/raw/train.jsonl \
  --valid_in data/raw/valid.jsonl \
  --train_out data/processed/train.jsonl \
  --valid_out data/processed/valid.jsonl \
  --min_conf 0.0
