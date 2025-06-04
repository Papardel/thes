#!/usr/bin/env python3
"""
Qwen-2.5-Coder-7B-Instruct runner – **prompt 3 (method-source)**

Writes <bug>_qwen_output_source.json for each bug-report JSON.
"""

from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

def load_qwen(*, device: Optional[str] = None,
              dtype: torch.dtype = torch.float16,
              full_gpu: bool = False
              ) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if full_gpu or device != "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=dtype, trust_remote_code=True).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=dtype, device_map="auto", trust_remote_code=True)

    model.eval()
    return tok, model, device

@torch.inference_mode()
def generate_with_qwen(
    prompts: Sequence[str],
    *,
    tokenizer,
    model,
    device,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    use_cache: bool = False,
) -> List[dict]:

    out: list[dict] = []
    for i, prompt in enumerate(prompts, 1):
        ids = tokenizer(prompt, return_tensors="pt").to(device)
        gen = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_cache=use_cache,
        )
        txt = tokenizer.decode(gen[0], skip_special_tokens=True)
        if txt.startswith(prompt):
            txt = txt[len(prompt):].strip()
        out.append({"prompt_index": i, "prompt": prompt, "response": txt})
    return out


from build_prompts_from_json import build_method_source_prompt, build_class_prompt, build_method_prompt
from pipeline import default_top5                              # five top sigs

def build_prompts(json_path: Path, tokenizer) -> List[str]:
    ctx  = tokenizer.model_max_length
    top5 = default_top5(json_path)
    #return build_class_prompt(json_path, tokenizer, ctx)
    return build_method_prompt(json_path, tokenizer, ctx)
    #return build_method_source_prompt(json_path, top5, tokenizer, ctx)


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

    prompts  = build_prompts(json_file, tokenizer)
    results  = generate_with_qwen(
        prompts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
    )

    out_root = output_dir or json_file.parent
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{json_file.stem}_qwen_output_source.json"

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"bug_id": json_file.stem, "results": results}, fh, indent=2)
    print(f"[saved] {out_path}")

def iter_json_files(folder: Path, recursive=False):
    pat = "**/*.json" if recursive else "*.json"
    yield from (p for p in folder.glob(pat) if p.is_file())

def main(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(
        prog="qwen-source-runner",
        description=("Run Qwen-2.5-Coder-7B-Instruct on prompt-3 "
                     "(method-source) JSON files"),
    )
    ap.add_argument("paths", nargs="+", help="JSON file(s) or dir(s)")
    ap.add_argument("--output_dir", help="Folder for all outputs")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--device", default=None, help="cuda / cpu / mps (auto)")
    ap.add_argument("--dtype", choices=["float16","bfloat16","float32"], default="float16")
    ap.add_argument("--full_gpu", action="store_true", help="Force full model on one GPU")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--use_cache", action="store_true")
    args = ap.parse_args(argv)

    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    tok, model, device = load_qwen(device=args.device, dtype=dtype, full_gpu=args.full_gpu)

    jsons: List[Path] = []
    for p in args.paths:
        pth = Path(p)
        if pth.is_dir():
            jsons.extend(iter_json_files(pth, recursive=args.recursive))
        elif pth.suffix == ".json":
            jsons.append(pth)
    if not jsons:
        ap.error("No JSON files found.")

    jsons.sort(key=lambda p: (p.stem.split("_")[0].lower(), int(p.stem.split("_")[1])))
    out_root = Path(args.output_dir) if args.output_dir else None

    for jf in jsons:
        print(jf.name)
        process_json_file(
            jf,
            tokenizer=tok,
            model=model,
            device=device,
            output_dir=out_root,
            max_new_tokens=args.max_new_tokens,
            use_cache=args.use_cache,
        )

if __name__ == "__main__":
    main()
