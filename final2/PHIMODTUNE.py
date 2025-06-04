#!/usr/bin/env python3
import os
import json
from pathlib import Path
from typing import Sequence, Optional, Tuple, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

################################################################################
# Model helpers
################################################################################

def load_phi3(
    *,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    """Load your fine-tuned Phi-3 model **once** and return (tokenizer, model, device)."""

    # pick device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ——— load tokenizer from your finetuned directory ———
    tokenizer = AutoTokenizer.from_pretrained(
        "models/phi3-buggy-class",   # <— point here to your --output-dir
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ——— load model (base + LoRA adapters) from the same dir ———
    model = AutoModelForCausalLM.from_pretrained(
        "models/phi3-buggy-class",   # <— point here as well
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    return tokenizer, model, device

################################################################################
# Generation helpers
################################################################################

@torch.inference_mode()
def generate_with_phi3(
    prompts: Sequence[str],
    *,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    use_cache: bool = False,
) -> List[dict]:
    """Sample one response per prompt."""
    results: list[dict] = []
    for idx, prompt in enumerate(prompts, start=1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_cache=use_cache,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        results.append({
            "prompt_index": idx,
            "prompt": prompt,
            "response": text,
        })
    return results

################################################################################
# JSON file helpers
################################################################################

def build_prompts(input_path: Path, tokenizer) -> List[str]:
    from final2.model_callers.build_prompts_from_json import build_class_prompt
    context_size = tokenizer.model_max_length
    return build_class_prompt(input_path, tokenizer, context_size)

def process_json_file(
    json_file: Path,
    *,
    tokenizer,
    model,
    device: str,
    output_dir: Optional[Path] = None,
    max_new_tokens: int = 512,
    use_cache: bool = False,
):
    prompts = build_prompts(json_file, tokenizer)
    results = generate_with_phi3(
        prompts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
    )
    out = output_dir or json_file.parent
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{json_file.stem}_phi3_output.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump({"bug_id": json_file.stem, "results": results}, f, indent=2)
    print(f"Wrote → {path}")

################################################################################
# Utilities
################################################################################

def iter_json_files(folder: Path, recursive: bool = False):
    pattern = "**/*.json" if recursive else "*.json"
    yield from (p for p in folder.glob(pattern) if p.is_file())

################################################################################
# CLI
################################################################################

def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="phi3-runner",
        description="Run your fine-tuned Phi-3 over bug-report JSON files (loads model once)",
    )
    parser.add_argument("paths", nargs="+", help="JSON file(s) or directory/ies")
    parser.add_argument("--output_dir", help="Where to write outputs (default: same folder)")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirs")
    parser.add_argument("--device", default=None, help="‘cuda’, ‘cpu’, or ‘mps’ (default: auto)")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model precision",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--use_cache", action="store_true", help="Enable KV-cache")

    args = parser.parse_args()
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # — load your fine-tuned model once —
    tokenizer, model, device = load_phi3(device=args.device, dtype=dtype)

    # — find all JSON inputs —
    all_jsons: List[Path] = []
    for p in args.paths:
        p = Path(p)
        if p.is_dir():
            all_jsons += list(iter_json_files(p, recursive=args.recursive))
        elif p.is_file() and p.suffix == ".json":
            all_jsons.append(p)
    if not all_jsons:
        parser.error("No JSON files found.")

    # sort by bug ID if named like “Project_1_…json”
    all_jsons.sort(key=lambda q: (q.stem.split("_")[0].lower(), int(q.stem.split("_")[1])))
    for jf in all_jsons:
        print(jf.name)

    common_out = Path(args.output_dir) if args.output_dir else None
    for jf in all_jsons:
        process_json_file(
            jf,
            tokenizer=tokenizer,
            model=model,
            device=device,
            output_dir=common_out,
            max_new_tokens=args.max_new_tokens,
            use_cache=args.use_cache,
        )

if __name__ == "__main__":
    main()
