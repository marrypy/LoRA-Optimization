import re

def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def drop_if_too_short(instruction: str, inp: str, out: str, min_chars: int = 20) -> bool:
    return (len(instruction) + len(inp) + len(out)) < min_chars

def basic_bad_output_filter(out: str) -> bool:
    bad = ["as an ai language model", "i can't browse the web", "cannot provide that"]
    lower = out.lower()
    return any(b in lower for b in bad)
