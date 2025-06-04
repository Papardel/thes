#!/usr/bin/env python3
"""
Run ONLY the “method-source” prompt (build_method_source_prompt) with Phi-3 and
write <bug>_phi3_output_source.json
"""
import argparse, json, os
from pathlib import Path
from typing import List, Sequence, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from build_prompts_from_json import build_class_prompt, build_method_prompt, build_method_source_prompt
from pipeline import default_top5         


def load_phi3(device: Optional[str] = None,
              dtype: torch.dtype = torch.float16
              ) -> Tuple["AutoTokenizer", "AutoModelForCausalLM", str]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    ).eval()
    return tok, model, device

@torch.inference_mode()
def sample_one(prompt: str,
               *,
               tok, model, device,
               max_new_tokens: int = 512,
               temperature: float = 0.7,
               top_p: float = 0.95,
               top_k: int = 50,
               use_cache: bool = True) -> str:

    tok_ids = tok(prompt, return_tensors="pt").to(device)
    if tok_ids.input_ids.size(1) > tok.model_max_length:
        raise ValueError(f"Prompt is {tok_ids.input_ids.size(1)} tokens "
                         f"(max {tok.model_max_length}).")

    out = model.generate(
        **tok_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        use_cache=use_cache,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip() if text.startswith(prompt) else text

def run_source_prompt(
    json_file: Path,
    *,
    tok, model, device,
    max_new_tokens: int,
    out_dir: Path,
    use_cache: bool):

    top5        = default_top5(json_file)
    #src_prompts = build_class_prompt(json_file, tok, tok.model_max_length)
    src_prompts = build_method_prompt(json_file, tok, tok.model_max_length)
    #src_prompts = build_method_source_prompt(json_file, top5, tok, tok.model_max_length)

    results: List[dict] = []
    for i, pr in enumerate(src_prompts, 1):
        resp = sample_one(pr,
                          tok=tok, model=model, device=device,
                          max_new_tokens=max_new_tokens,
                          use_cache=use_cache)
        results.append({"prompt_index": i, "prompt": pr, "response": resp})

    project, bug_id = json_file.stem.split("_", 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{project}_{bug_id}_phy3_output.json"
    out_path.write_text(json.dumps({"bug_id": bug_id, "results": results},
                                   indent=2), encoding="utf-8")
    print(f"✔  {json_file.name} → {out_path.name}")

def main(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(prog="phi3-source-only")
    p.add_argument("paths", nargs="+", help="JSON file(s) or dir(s)")
    p.add_argument("--output_dir", default=None,
                   help="Where to put *_phi3_output_source.json (default: next to each file)")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["float16","bfloat16","float32"],
                   default="float16")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--use_cache", action="store_true")
    args = p.parse_args(argv)

    dtype_map = {"float16": torch.float16,
                 "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    tok, model, device = load_phi3(device=args.device, dtype=dtype_map[args.dtype])

    targets: List[Path] = []
    for pth in args.paths:
        pth = Path(pth)
        if pth.is_dir():
            pat = "**/*.json" if args.recursive else "*.json"
            targets.extend(p for p in pth.glob(pat) if p.is_file())
        elif pth.suffix == ".json":
            targets.append(pth)
    if not targets:
        p.error("No JSON files found.")

    out_root = Path(args.output_dir) if args.output_dir else None
    for jf in sorted(targets):
        run_source_prompt(
            jf,
            tok=tok, model=model, device=device,
            max_new_tokens=args.max_new_tokens,
            out_dir=(out_root or jf.parent),
            use_cache=args.use_cache
        )

if __name__ == "__main__":
    main()
