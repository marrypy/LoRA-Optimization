#!/usr/bin/env bash
set -e
python -m src.train.sft_lora --config configs/sft_qwen_0_5b.yaml
