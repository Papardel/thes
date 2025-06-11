import re
import json
from pathlib import Path
from typing import List

FQCN_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")

def _looks_like_fqcn(token: str) -> bool:
    return bool(FQCN_RE.fullmatch(token.strip("`").rstrip(" ,;.")))


def _find_response_marker(lines: List[str]) -> int | None:

    for i in reversed(range(len(lines))):
        stripped = lines[i].lstrip("#").strip()
        if stripped.lower().startswith("response:"):
            return i
    return None

def extract_and_validate_response(output: str) -> List[str]:
    lines = [ln.rstrip() for ln in output.splitlines() if ln.strip()]
    idx = _find_response_marker(lines)

    if idx is not None:
        after = lines[idx].split(":", 1)[1].strip()
        search_area = ([after] if after else []) + lines[idx + 1 :]
    else:
        search_area = lines

    seen, fqcn_list = set(), []
    for tok in search_area:
        if tok.lower().startswith("assistant"):
            continue
        candidate = tok.strip("`").rstrip(" ,;.")
        if _looks_like_fqcn(candidate) and candidate not in seen:
            seen.add(candidate)
            fqcn_list.append(candidate)

    if not fqcn_list:
        print("No plausible class names found in output.")
    return fqcn_list


def check_class_response(output: str, json_path: Path) -> List[str]:
    """
    Return the subset of class names from `extract_and_validate_response`
    that actually exist in the project’s JSON metadata.
    """
    fqcn_list = extract_and_validate_response(output)
    if not fqcn_list:
        return []

    data = json.loads(json_path.read_text(encoding="utf-8"))
    valid = {
        cls["name"].replace("/", ".")[:-5]       
        for cls in data.get("classes", [])
    }

    result = [c for c in fqcn_list if c in valid]
    missing = set(fqcn_list) - set(result)
    for m in missing:
        print(f"Class '{m}' not found in JSON classes list.")
    return result

_JAVA_MODIFIERS = (
    "public", "protected", "private",
    "static", "final", "abstract", "native", "synchronized",
    "transient", "volatile", "strictfp",
)

_MOD_RX = re.compile(rf"^(?:{'|'.join(_JAVA_MODIFIERS)})\s+")

def _normalize_signature(sig: str) -> str:
    sig = sig.strip("`").lstrip("-*# ").strip()
    while True:
        new_sig = _MOD_RX.sub("", sig, count=1)
        if new_sig == sig:
            break
        sig = new_sig.strip()
    sig = re.sub(r"\s+", " ", sig)
    return sig

def check_rank_response(output: str, json_path: Path) -> List[str]:
    lines = [ln.rstrip() for ln in output.splitlines()]
    marker_idx = next(
        (i for i, ln in enumerate(lines)
         if ln.lstrip("#").strip().lower().startswith("response:")),
        None,
    )
    if marker_idx is None:
        print("No 'RESPONSE:' line found in rank output.")
        return []

    raw_sigs = [
        ln.split(":", 1)[0].lstrip("-*# ").strip()
        for ln in lines[marker_idx + 1 :]
        if ln.strip()
    ]

    data = json.loads(json_path.read_text(encoding="utf-8"))
    buggy_norm = {
        _normalize_signature(sig)
        for cls in data.get("classes", [])
        for sig in cls.get("buggy_signatures", [])
    }

    seen, result = set(), []
    for raw in raw_sigs:
        norm = _normalize_signature(raw)
        if norm in buggy_norm and raw not in seen:
            seen.add(raw)
            result.append(raw)

    return result