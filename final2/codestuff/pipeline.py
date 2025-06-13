#!/usr/bin/env python3
"""
pipeline.py

End-to-end script:

1. Build a **class** prompt → run model → validate → log.
2. Build **method-ranking** prompts for ONLY the validated classes →
   run model → validate → log.
3. Build **source-inspection** prompts for the accepted buggy methods →
   run model → extract most-likely lines → log.

Every stage appends its response to `pipeline_responses.txt`.
"""

from __future__ import annotations

import re
import sys
import json
import random
import shutil
from pathlib import Path
from typing import List

from retrieval.cli import get_context
from utils.prompts.builders.build_prompts_from_json import (
    build_class_prompt,
    build_method_prompt,
    build_method_source_prompt,
)
from utils.prompts.validators.input_normalizer import (
    check_class_response,
    check_rank_response,
    extract_line_response,
)
from LLM.diagnosis import infer_base
from LLM.modelLoader import ModelLoader


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────
def _parse_method_signature(src: str) -> str:
    for ln in src.splitlines():
        ln = ln.strip()
        if ln:
            return ln.rstrip("{").strip()
    raise ValueError("Empty method body provided")


# ──────────────────────────────────────────────────────────────
#  Prompt builders (wrappers around utils.builders)
# ──────────────────────────────────────────────────────────────
def getClass(json_path: Path, tokenizer, ctx_limit: int, outdir: Path) -> Path:
    prompt = build_class_prompt(json_path, tokenizer, ctx_limit)[0]
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / f"{json_path.stem}_class.txt"
    dst.write_text(prompt, encoding="utf-8")
    print(f"Generated class prompt for {json_path.stem}")
    return dst


def getRank(
    json_path: Path,
    tokenizer,
    ctx_limit: int,
    outdir: Path,
    class_names: List[str] | None = None,
) -> List[Path]:
    """
    Build ranking prompts, **filtering** so that each prompt references
    at least one of the validated `class_names` (if provided).
    """
    prompts = build_method_prompt(json_path, tokenizer, ctx_limit)
    outdir.mkdir(parents=True, exist_ok=True)

    if class_names:
        # keep only the prompt chunks that mention one of the classes
        filtered = []
        for pr in prompts:
            if any(re.search(rf"\b{re.escape(c)}\b", pr) for c in class_names):
                filtered.append(pr)
        if filtered:
            prompts = filtered  # fallback to all if filter emptied list

    paths: List[Path] = []
    if len(prompts) == 1:
        p = outdir / f"{json_path.stem}_rank.txt"
        p.write_text(prompts[0], encoding="utf-8")
        paths.append(p)
    else:
        for idx, pr in enumerate(prompts, 1):
            p = outdir / f"{json_path.stem}_rank_{idx}.txt"
            p.write_text(pr, encoding="utf-8")
            paths.append(p)

    print(f"Generated {len(paths)} ranking prompts for {json_path.stem}")
    return paths


def getSource(
    json_path: Path,
    tokenizer,
    ctx_limit: int,
    outdir: Path,
    sigs: List[str],
) -> List[Path]:
    if not sigs:
        raise ValueError("No method signatures provided for source prompt.")
    prompts = build_method_source_prompt(json_path, sigs, tokenizer, ctx_limit)
    outdir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    if len(prompts) == 1:
        p = outdir / f"{json_path.stem}_source.txt"
        p.write_text(prompts[0], encoding="utf-8")
        paths.append(p)
    else:
        for idx, pr in enumerate(prompts, 1):
            p = outdir / f"{json_path.stem}_source_{idx}.txt"
            p.write_text(pr, encoding="utf-8")
            paths.append(p)

    print(f"Generated {len(paths)} source prompts for {json_path.stem}")
    return paths


def cleanup(buggy_dir: Path, fixed_dir: Path) -> None:
    shutil.rmtree(buggy_dir)
    shutil.rmtree(fixed_dir)

# ──────────────────────────────────────────────────────────────
#  Main pipeline
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("Usage: pipeline.py <MODEL> <PROJECT> <BUG_ID> <OUT> ['train full']")

    model_name, project, bug_id, out_base, *flag_parts = sys.argv[1:]
    flags = " ".join(flag_parts)

    buggy_dir, fixed_dir = get_context(project, bug_id, out_base, flags)

    try:
        loader = ModelLoader(model_name)
        tokenizer, model, device = loader.loadModel()
        max_ctx = tokenizer.model_max_length
    except Exception as e:
        sys.exit(f"Error loading model: {e}")

    json_path = Path(f"{out_base}.json").resolve()
    outdir = Path(out_base).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    LOG_PATH = outdir / "pipeline_responses.txt"

    class_prompt = getClass(json_path, tokenizer, max_ctx, outdir)
    class_resp_path = infer_base(
        "class", class_prompt, model, tokenizer, device, max_ctx, model_name
    )
    class_resp, ok = check_class_response(
        class_resp_path.read_text(encoding="utf-8"),
        json_path,
        log_path=LOG_PATH,
    )
    print(f"Validated classes: {class_resp}")
    if( not ok):
        cleanup(buggy_dir, fixed_dir)
        sys.exit("No valid classes found. Exiting pipeline.")
    

    rank_prompts = getRank(json_path, tokenizer, max_ctx, outdir, class_resp)
    rank_resp_path = infer_base(
        "rank", rank_prompts, model, tokenizer, device, max_ctx, model_name
    )
    rank_resp = check_rank_response(
        rank_resp_path.read_text(encoding="utf-8"),
        json_path,
        log_path=LOG_PATH,
    )
    print(f"Validated buggy signatures: {rank_resp}")

    try:
        source_prompts = getSource(json_path, tokenizer, max_ctx, outdir, rank_resp)
        source_resp_path = infer_base(
            "source", source_prompts, model, tokenizer, device, max_ctx, model_name
        )
        line_resp = extract_line_response(
            source_resp_path.read_text(encoding="utf-8"), log_path=LOG_PATH
        )
        print(f"Line response: {line_resp}")
    except ValueError as e:
        sys.exit(f"Error generating source prompt: {e}")

    cleanup(buggy_dir, fixed_dir)
