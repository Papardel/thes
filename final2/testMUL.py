#!/usr/bin/env python3
"""Fine‑tune LoRA adapters **without** bitsandbytes quantisation.
Compatible with single‑GPU or multi‑GPU (DDP/FSDP) runs on A100‑class
hardware using FP16/BF16 precision.
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("agent_id", type=int, choices=[1, 2, 3])
parser.add_argument("base_model_name", help="HF model id or local path")
parser.add_argument(
    "--bf16", action="store_true", help="Force BF16 even if CUDA says no",
)
cli = parser.parse_args()

AGENT_ID = cli.agent_id
BASE_MODEL = cli.base_model_name
modname = BASE_MODEL.split("/")[-1]
# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_DIR = "./data"
OUTPUT_DIR = "./adapters/10"
JSONL_PATH = os.path.join(DATA_DIR, f"agent{AGENT_ID}.train.jsonl")
os.makedirs(OUTPUT_DIR, exist_ok=True)
assert os.path.exists(JSONL_PATH), f"Missing dataset file: {JSONL_PATH}"

MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH = 2048, 256

TRAIN_ARGS = dict(
    learning_rate=5e-5,
    per_device_train_batch_size=8,  # adjust at runtime if needed
    gradient_accumulation_steps=2,
    num_train_epochs=25,
    bf16=torch.cuda.is_available() or cli.bf16,
    fp16=not torch.cuda.is_available(),
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=500,
    dataloader_num_workers=8,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_target_modules(model):
    names = [n for n, _ in model.named_modules()]
    if any("qkv_proj" in n for n in names):
        return ["qkv_proj", "o_proj"]
    if {"q_proj", "v_proj"} <= {n.split(".")[-1] for n in names}:
        return ["q_proj", "v_proj", "o_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def preprocess_causal(ex, tok):
    ids_in = tok(ex["prompt"], truncation=True, max_length=MAX_SOURCE_LENGTH)["input_ids"]
    ids_out = tok(ex["response"], truncation=True, max_length=MAX_TARGET_LENGTH)["input_ids"]
    eos = tok.eos_token_id
    full = (ids_in + [eos] + ids_out)[: MAX_SOURCE_LENGTH + MAX_TARGET_LENGTH]

    pad_len = MAX_SOURCE_LENGTH + MAX_TARGET_LENGTH - len(full)
    full += [tok.pad_token_id] * pad_len
    attn = [1] * (len(full) - pad_len) + [0] * pad_len

    try:
        eos_pos = full.index(eos)
    except ValueError:
        eos_pos = len(ids_in)

    labels = [-100 if i <= eos_pos else full[i] for i in range(len(full))]
    return {"input_ids": full, "attention_mask": attn, "labels": labels}


def compute_metrics(_, raw_valid, tok, mdl):
    mdl.eval()
    hits = 0
    with torch.no_grad():
        for ex in raw_valid:
            enc = tok(
                ex["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SOURCE_LENGTH,
            ).to(mdl.device)
            out = mdl.generate(
                **enc,
                max_new_tokens=MAX_TARGET_LENGTH,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )[0]
            eos_pos = (out == tok.eos_token_id).nonzero(as_tuple=True)[0]
            gen = tok.decode(
                out[eos_pos[0] + 1 :] if len(eos_pos) else out,
                skip_special_tokens=True,
            ).strip()
            hits += gen == ex["response"].strip()
    return {"exact_match": hits / len(raw_valid)}

# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(agent_id: int, base: str):
    adapter_name = f"agent_{modname}_{agent_id}_lora"
    print(f"\nTraining {adapter_name} on {base}\n")

    # Dataset
    raw = load_dataset("json", data_files=JSONL_PATH)["train"]
    split = raw.train_test_split(test_size=0.1, seed=42)
    train_raw, valid_raw = split["train"], split["test"]

    # Tokeniser
    tok = AutoTokenizer.from_pretrained(base, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Precision choice
    dtype = torch.bfloat16 if (torch.cuda.is_available() and TRAIN_ARGS["bf16"]) else torch.float16

    # Backbone (full precision, no quantisation)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        local_files_only=True,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # LoRA
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=find_target_modules(model),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Tokenised datasets
    train_ds = train_raw.map(lambda ex: preprocess_causal(ex, tok), remove_columns=["prompt", "response"])
    valid_ds = valid_raw.map(lambda ex: preprocess_causal(ex, tok), remove_columns=["prompt", "response"])

    # TrainingArguments
    args = TrainingArguments(
        output_dir=f"./results/{adapter_name}",
        logging_dir=f"./logs/{adapter_name}",
        **TRAIN_ARGS,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tok,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, valid_raw, tok, model),
    )

    trainer.train()
    model.save_pretrained(os.path.join(OUTPUT_DIR, adapter_name))
    print(f"Saved adapter to {OUTPUT_DIR}/{adapter_name}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train(AGENT_ID, BASE_MODEL)
    print("Completed.")

