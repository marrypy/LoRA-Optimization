# Sci-LLM Robustness with LoRA

A fully reproducible machine learning project that improves the reliability of open-weight large language models using data engineering, LoRA fine-tuning, and rigorous evaluation. Designed to run locally on Apple Silicon : ) (proud mac user here ik)

---

## 🚀 Project Goal

This project demonstrates how to:

- clean and structure noisy instruction data  
- fine-tune a pretrained open-weight LLM using LoRA adapters  
- evaluate model improvement using real benchmarks  
- build a reliable, production-style ML pipeline  

Rather than training a model from scratch, the project improves the behavior of an existing LLM and proves that the improvement is real.

--- 

This mirrors real workflows used in ML research labs and applied AI teams.

---

## Quickstart

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Add data (or generate sample data, feel free to tweak this)
```bash 
scripts/0_download_example_data.sh
```

### 3) Build dataset
```bash
scripts/1_build_dataset.sh
```

### 4) Train (LoRA on small model)
```bash 
scripts/2_train.sh
```

### 5) Evaluate
```bash
bash scripts/3_eval.sh
```

### Notes:
- Defaults to a small model for local training
- You can swap models in configs/
- Small example datasets are included in data/raw/ for reproducibility