from __future__ import annotations
from retrieval.cli import get_context
import sys
import json
import random, shutil
from pathlib import Path
from typing import List
from utils.prompts.builders.build_prompts_from_json import build_class_prompt, build_method_prompt, build_method_source_prompt
from LLM.diagnosis import infer_base
from LLM.modelLoader import ModelLoader
from utils.prompts.validators.input_normalizer import check_class_response, check_rank_response

def _parse_method_signature(method_src: str) -> str:
    """Return the *declaration line* (no opening brace) of a Java method."""
    for line in method_src.splitlines():
        line = line.strip()
        if line:
            return line.rstrip("{").strip()
    raise ValueError("Empty method body provided")


def default_top5(json_path: Path) -> List[str]:
    """Pick one *buggy* method + four random others from `buggy_signatures`."""

    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    classes = data.get("classes", [])
    if not classes:
        raise ValueError("No classes found in JSON")

    primary_sig = None
    for cls in classes:
        methods = cls.get("methods") or []
        if not methods:
            continue
        buggy_src = methods[0]["buggy_method"]
        simple_sig = _parse_method_signature(buggy_src)
        fqcn = cls["name"].replace("/", ".").rstrip(".java")
        primary_sig = f"{fqcn}.{simple_sig}"
        candidates = cls.get("buggy_signatures") or []
        break

    if primary_sig is None:
        return default(classes)

    def _simple(s: str) -> str:
        return s.split(None, 1)[-1]

    remaining = [s for s in candidates if _simple(s) not in primary_sig]
    if len(remaining) < 4:
        pool = (remaining or candidates) * 5
    else:
        pool = remaining
    other_sigs = random.sample(pool, 4)

    return [primary_sig] + [f"{fqcn}.{s}" for s in other_sigs]


def default(classes) -> List[str]:
    """Original behaviour: first five `buggy_signatures` of the first class."""
    for cls in classes:
        sigs = cls.get("buggy_signatures") or []
        if sigs:
            fqcn = cls["name"].replace("/", ".").rstrip(".java")
            return [f"{fqcn}.{s}" for s in sigs[:5]]
    raise ValueError("No signatures found in JSON – cannot build source prompt")

def getClass(json_path: Path, tokenizer, ctx_limit: int, outdir: Path) -> Path:
    prompt = build_class_prompt(json_path, tokenizer, ctx_limit)[0]
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"{json_path.stem}_class.txt"
    dest.write_text(prompt, encoding="utf-8")
    print(f"Generated class prompt for {json_path.stem}")
    return dest

def getRank(json_path: Path, tokenizer, ctx_limit: int, outdir: Path) -> List[Path]:
    prompts = build_method_prompt(json_path, tokenizer, ctx_limit)
    outdir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    if len(prompts) == 1:
        pth = outdir / f"{json_path.stem}_rank.txt"
        pth.write_text(prompts[0], encoding="utf-8")
        paths.append(pth)
    else:
        for idx, pr in enumerate(prompts, 1):
            pth = outdir / f"{json_path.stem}_rank_{idx}.txt"
            pth.write_text(pr, encoding="utf-8")
            paths.append(pth)

    print(f"Generated {len(paths)} ranking prompts for {json_path.stem}")
    return paths

def getSource(json_path: Path, tokenizer, ctx_limit: int, outdir: Path,
              top5: List[str] | None = None) -> List[Path]:
    sigs = top5 or default_top5(json_path)
    prompts = build_method_source_prompt(json_path, sigs, tokenizer, ctx_limit)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    if len(prompts) == 1:
        pth = outdir / f"{json_path.stem}_source.txt"
        pth.write_text(prompts[0], encoding="utf-8")
        paths.append(pth)
    else:
        for idx, p in enumerate(prompts, 1):
            pth = outdir / f"{json_path.stem}_source_{idx}.txt"
            pth.write_text(p, encoding="utf-8")
            paths.append(pth)
    print(f"Generated {len(paths)} source prompts for {json_path.stem}")
    return paths

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("Usage: pipeline.py <MODEL> <PROJECT> <BUG_ID> <OUT> ['train full']")
    loader = None
    modeName, project, bug_id, out_base, *flag_parts = sys.argv[1:]
    flags = " ".join(flag_parts)            
    buggy, fixed = get_context(project, bug_id, out_base, flags)
    try:        
        loader = ModelLoader(modeName)
        tok, model, device  = loader.loadModel()
        max_ctx = tok.model_max_length
        json_path     = Path(f"{out_base}.json").resolve()   
        outdir        = Path(out_base).resolve()            
        outdir.mkdir(parents=True, exist_ok=True)
        #for aid in (1, 2, 3):
            #loader.attach_adapter(model, aid) 
    except Exception as e:
        sys .exit(f"Error setting up pipeline: {e}")
    class_resp = None
    rank_resp = None


    class_file = getClass(json_path, tok, max_ctx, outdir)
    #model = loader.attach_adapter(model, 1)
    class_resp_path = infer_base("class", class_file, model, tok, device, max_ctx, modeName) # LLM
    class_resp = check_class_response(class_resp_path.read_text(encoding="utf-8"), json_path) # checkl
    print(f"Class response: {class_resp}")
    
    rank_file = getRank(json_path, tok, max_ctx, outdir)
    #model = loader.attach_adapter(model, 2)
    rank_resp_path = infer_base("rank", rank_file, model, tok, device, max_ctx, modeName)

    rank_resp = check_rank_response(rank_resp_path.read_text(encoding="utf-8"), json_path)
    print(f"Rank response: {rank_resp}")

    source_file = getSource(json_path, tok, max_ctx, outdir, rank_resp)
    #model = loader.attach_adapter(model, 3)
    infer_base("source", source_file, model, tok, device, max_ctx, modeName)
    
    shutil.rmtree(buggy)
    shutil.rmtree(fixed)

