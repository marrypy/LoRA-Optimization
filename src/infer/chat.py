import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.train.utils import pick_device

def generate(model_dir: str, prompt: str, max_new_tokens: int = 200) -> str:
    device = pick_device()

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if device in ["cuda","mps"] else torch.float32,
    ).to(device)

    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
    return tok.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--prompt", required=True)
    args = ap.parse_args()

    print(generate(args.model_dir, args.prompt))
