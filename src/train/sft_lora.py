import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from transformers import Trainer, DataCollatorForLanguageModeling


from src.data.schema import Example
from .utils import pick_device

def load_jsonl(path: str):
    return load_dataset("json", data_files=path, split="train")

def format_example(ex) -> str:
    e = Example(instruction=ex["instruction"], input=ex.get("input",""), output=ex["output"])
    return e.to_prompt() + e.output.strip()

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    model_name = cfg["model_name"]
    train_path = cfg["dataset_train"]
    valid_path = cfg["dataset_valid"]
    max_seq_len = int(cfg.get("max_seq_len", 512))

    device = pick_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
    )
    model.config.use_cache = False

    # LoRA setup
    lora_cfg = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)

    # Move model
    model.to(device)

    train_ds = load_jsonl(train_path)
    valid_ds = load_jsonl(valid_path)

    train_ds = train_ds.map(lambda x: {"text": format_example(x)}, remove_columns=train_ds.column_names)
    valid_ds = valid_ds.map(lambda x: {"text": format_example(x)}, remove_columns=valid_ds.column_names)

    tcfg = cfg["training"]
    args = TrainingArguments(
        output_dir=tcfg["output_dir"],
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        num_train_epochs=float(tcfg["num_train_epochs"]),
        learning_rate=float(tcfg["learning_rate"]),
        warmup_ratio=float(tcfg["warmup_ratio"]),
        logging_steps=int(tcfg["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(tcfg["eval_steps"]),
        save_steps=int(tcfg["save_steps"]),
        optim=tcfg.get("optim", "adamw_torch"),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        report_to="none",
    )

    # Tokenize datasets for standard Trainer
    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    valid_tok = valid_ds.map(tokenize, batched=True, remove_columns=valid_ds.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        data_collator=data_collator,
    )

    trainer.train()


    trainer.train()
    trainer.save_model(tcfg["output_dir"])
    tokenizer.save_pretrained(tcfg["output_dir"])
    print(f"Saved to: {tcfg['output_dir']}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
