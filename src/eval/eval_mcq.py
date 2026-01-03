import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.train.utils import pick_device
from .eval_calibration import expected_calibration_error

def load_mcq(path: str):
    """
    Each line:
    {"question": "...", "choices": ["A..","B..","C..","D.."], "answer_index": 2}
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def score_choice(model, tok, device, prompt: str) -> float:
    """
    Score next-token probability for a choice label like "A", "B", ...
    We make it simple: ask model to answer with a single letter.
    """
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0, -1]  # next token logits
        probs = torch.softmax(logits, dim=-1)
    return probs

def eval_mcq(model_dir: str, mcq_path: str):
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if device in ["cuda","mps"] else torch.float32,
    ).to(device)

    items = load_mcq(mcq_path)
    labels = []
    confs = []
    corrects = []

    # Token ids for "A", "B", "C", "D"
    label_tokens = {k: tok.encode(k, add_special_tokens=False)[0] for k in ["A","B","C","D"]}

    for it in items:
        q = it["question"]
        choices = it["choices"]
        ans = int(it["answer_index"])

        prompt = (
            "Answer with only one letter: A, B, C, or D.\n\n"
            f"Question: {q}\n"
            f"A) {choices[0]}\n"
            f"B) {choices[1]}\n"
            f"C) {choices[2]}\n"
            f"D) {choices[3]}\n"
            "Answer:"
        )

        probs = score_choice(model, tok, device, prompt)
        p = np.array([float(probs[label_tokens[k]].cpu()) for k in ["A","B","C","D"]])
        pred = int(p.argmax())
        conf = float(p.max())

        correct = 1 if pred == ans else 0
        corrects.append(correct)
        confs.append(conf)

    acc = float(np.mean(corrects))
    ece = expected_calibration_error(np.array(confs), np.array(corrects))
    return acc, ece, len(items)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--mcq_path", required=True)
    args = ap.parse_args()

    acc, ece, n = eval_mcq(args.model_dir, args.mcq_path)
    print(f"N={n}  Accuracy={acc:.3f}  ECE={ece:.3f}")
