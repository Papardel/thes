#!/usr/bin/env python3
"""
Run *prompt 3* (build_method_source_prompt) with DeepSeek-Coder and write
<bug>_deepseek_output_source.json.
"""
import os, json
from pathlib import Path
from typing import Sequence, Iterable, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"

def load_deepseek_coder(
    *,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
) -> Tuple["AutoTokenizer", "AutoModelForCausalLM", str]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    ).eval().to(device)

    return tokenizer, model, device

@torch.inference_mode()
def generate_with_deepseek(
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
        results.append({"prompt_index": idx, "prompt": prompt, "response": text})
    return results

from build_prompts_from_json import build_class_prompt, build_method_source_prompt, build_method_prompt   # prompt-3
from pipeline import default_top5                                # five sigs

def build_prompts(input_path: Path, tokenizer) -> List[str]:
    """Return prompt 3 strings that fit the model context."""
    ctx = tokenizer.model_max_length
    top5 = default_top5(input_path)
    #return build_class_prompt(input_path, tokenizer, ctx)
    #return build_method_prompt(input_path, tokenizer, ctx)
    return build_method_source_prompt(input_path, top5, tokenizer, ctx)

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

    results = generate_with_deepseek(
        prompts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
    )

    output_dir = output_dir or json_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{json_file.stem}_deepseek_output_source.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"bug_id": json_file.stem, "results": results}, f, indent=2)
    print(f"Wrote → {out_path}")

def iter_json_files(folder: Path, recursive: bool = False):
    pattern = "**/*.json" if recursive else "*.json"
    yield from (p for p in folder.glob(pattern) if p.is_file())

def main(argv: Optional[Sequence[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="deepseek-source-runner",
        description=("Run deepseek-ai/deepseek-coder-6.7b-instruct over "
                     "prompt 3 (method-source) JSON files; the model is loaded once."),
    )
    parser.add_argument("paths", nargs="+", help="JSON file(s) or directory/ies")
    parser.add_argument("--output_dir", help="Common output directory (default: alongside each JSON)")
    parser.add_argument("--recursive", action="store_true", help="Recurse into sub-directories when a path is a dir")
    parser.add_argument("--device", default=None, help="Force 'cuda', 'cpu', or 'mps' (default: auto)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Model precision")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens per prompt")
    parser.add_argument("--use_cache", action="store_true", help="Enable KV-cache if transformers ≥ 4.40")

    args = parser.parse_args(argv)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    tokenizer, model, device = load_deepseek_coder(device=args.device, dtype=dtype_map[args.dtype])

    all_jsons: List[Path] = []
    for p in args.paths:
        pth = Path(p)
        if pth.is_dir():
            all_jsons.extend(iter_json_files(pth, recursive=args.recursive))
        elif pth.suffix == ".json":
            all_jsons.append(pth)
    if not all_jsons:
        parser.error("No JSON files found among supplied paths.")

    all_jsons.sort(key=lambda p: (p.stem.split("_")[0].lower(), int(p.stem.split("_")[1])))
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
