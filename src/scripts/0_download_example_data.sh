#!/usr/bin/env bash
set -e

mkdir -p data/raw

cat > data/raw/train.jsonl << 'EOF'
{"instruction":"Answer clearly.","input":"What is overfitting?","output":"Overfitting is when a model learns patterns that don't generalize, performing well on training data but worse on new data."}
{"instruction":"Answer in 2 bullet points.","input":"How can you reduce overfitting?","output":"- Use regularization (dropout/weight decay)\n- Increase data or use augmentation"}
{"instruction":"Answer concisely.","input":"What is calibration in ML?","output":"Calibration measures whether predicted confidence matches actual correctness."}
{"instruction":"Answer with one sentence.","input":"Define inference for an LLM.","output":"Inference is using a trained model to generate outputs from inputs without updating its weights."}
EOF

cat > data/raw/valid.jsonl << 'EOF'
{"instruction":"Answer clearly.","input":"What is underfitting?","output":"Underfitting is when a model is too simple to capture patterns, performing poorly on both training and test data."}
{"instruction":"Answer with 1 sentence.","input":"What does LoRA do?","output":"LoRA trains small low-rank adapter matrices so you can fine-tune a model efficiently without updating all weights."}
EOF

# Optional simple MCQ eval set
mkdir -p data/raw
cat > data/raw/mcq.jsonl << 'EOF'
{"question":"Which best describes overfitting?","choices":["Model performs well on new data","Model learns noise and fails to generalize","Model has too few parameters","Model always underestimates loss"],"answer_index":1}
{"question":"What is inference?","choices":["Training weights","Evaluating by updating weights","Running the model to get outputs","Collecting more data"],"answer_index":2}
EOF

echo "Wrote: data/raw/train.jsonl, data/raw/valid.jsonl, data/raw/mcq.jsonl"
