import json
from pathlib import Path
from tqdm import tqdm

from .clean import normalize_whitespace, drop_if_too_short, basic_bad_output_filter
from .weak_label_filter import keep_by_confidence

def process_file(in_path: Path, out_path: Path, min_conf: float = 0.65):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept, dropped = 0, 0

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc=f"Processing {in_path.name}"):
            ex = json.loads(line)

            if not keep_by_confidence(ex, min_conf=min_conf):
                dropped += 1
                continue

            instruction = normalize_whitespace(ex["instruction"])
            inp = normalize_whitespace(ex.get("input", ""))
            out = normalize_whitespace(ex["output"])

            if drop_if_too_short(instruction, inp, out):
                dropped += 1
                continue
            if basic_bad_output_filter(out):
                dropped += 1
                continue

            f_out.write(json.dumps({
                "instruction": instruction,
                "input": inp,
                "output": out,
            }, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Kept: {kept}, Dropped: {dropped} -> {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", default="data/raw/train.jsonl")
    ap.add_argument("--valid_in", default="data/raw/valid.jsonl")
    ap.add_argument("--train_out", default="data/processed/train.jsonl")
    ap.add_argument("--valid_out", default="data/processed/valid.jsonl")
    ap.add_argument("--min_conf", type=float, default=0.65)
    args = ap.parse_args()

    process_file(Path(args.train_in), Path(args.train_out), min_conf=args.min_conf)
    process_file(Path(args.valid_in), Path(args.valid_out), min_conf=args.min_conf)
