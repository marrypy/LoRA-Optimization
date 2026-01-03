#!/usr/bin/env bash
set -e
python -m src.eval.eval_mcq \
  --model_dir runs/qwen_0_5b_sft_lora \
  --mcq_path data/raw/mcq.jsonl
chmod +x scripts/*.sh
