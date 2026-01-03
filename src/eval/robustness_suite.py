import re

def add_noise(s: str) -> str:
    # simple noise: extra spaces and random punctuation patterns
    s = re.sub(r"\s+", " ", s)
    return s.replace("?", "??").replace(".", "...")

def robustness_prompts():
    base = [
        "Explain overfitting in one sentence.",
        "List 3 ways to reduce hallucinations.",
        "Write a short JSON object with keys: a, b, c."
    ]
    noisy = [add_noise(x) for x in base]
    longer = [x + " " + "Add detail. " * 20 for x in base]
    return {"base": base, "noisy": noisy, "long": longer}
