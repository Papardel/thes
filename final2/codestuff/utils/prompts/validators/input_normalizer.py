"""
validators/input_normalizer.py

• Stage-1  → extract fully-qualified class names, validate against JSON, log.
• Stage-2  → extract buggy method signatures, ignore modifiers, validate, log.
• Stage-3  → extract "<signature>: line of code" pairs from the source prompt
             and log them (no validation yet — they’ll feed a follow-up step).

Each call appends its results (in order of execution) to a shared text
log if a `log_path` is supplied.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

# ──────────────────────────────────────────────────────────────
#  Logging helper
# ──────────────────────────────────────────────────────────────
def _append_to_log(step: str, items: List[str], log_path: Optional[Path]) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"=== {step.upper()} RESPONSE ===\n")
        fh.write("\n".join(items) + ("\n" if items else "(no items)\n"))
        fh.write("\n")  # blank-line separator


# ──────────────────────────────────────────────────────────────
#  Shared helpers
# ──────────────────────────────────────────────────────────────
FQCN_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")

def _looks_like_fqcn(tok: str) -> bool:
    return bool(FQCN_RE.fullmatch(tok.strip("`").rstrip(" ,;.")))

def _find_response_marker(lines: List[str]) -> int | None:
    for i in reversed(range(len(lines))):
        if lines[i].lstrip("#").strip().lower().startswith("response:"):
            return i
    return None

def extract_and_validate_response(output: str) -> Tuple[List[str], bool]:
    """
    Return a tuple **(fqcn_list, found)**
    • fqcn_list – every fully-qualified class name we spotted
    • found      – False when no class was discovered, True otherwise
    """
    lines = [ln.rstrip() for ln in output.splitlines() if ln.strip()]
    idx   = _find_response_marker(lines)
    search = ([lines[idx].split(":", 1)[1].strip()]
              if idx is not None and ":" in lines[idx] else []) + \
             (lines[idx + 1:] if idx is not None else lines)

    seen, fqcn = set(), []
    for tok in search:
        if tok.lower().startswith("assistant"):
            continue
        cand = tok.strip("`").rstrip(" ,;.")
        if _looks_like_fqcn(cand) and cand not in seen:
            seen.add(cand)
            fqcn.append(cand)

    if not fqcn:
        print("No plausible class names found in output.")
        return [], False                     # ← change

    return fqcn, True                        # ← change


def check_class_response(output: str, json_path: Path,
                         log_path: Optional[Path] = None) -> Tuple[List[str], bool]:
    """
    • Returns (valid_classes, found)
      - *valid_classes* – subset of the extracted classes that really exist
      - *found*         – False when **no** class was extracted at all,
                          True otherwise (regardless of later validation)
    """
    cand, found = extract_and_validate_response(output)   # ← unpack tuple
    if not found:                                         # ← change
        _append_to_log("class", [], log_path)
        return [], False                                  # ← change

    # validate against JSON
    valid = {cls["name"].replace("/", ".")[:-5]
             for cls in json.loads(json_path.read_text()).get("classes", [])}
    res = [c for c in cand if c in valid]
    for miss in set(cand) - set(res):
        print(f"Class '{miss}' not found in JSON classes list.")

    _append_to_log("class", cand, log_path)
    return res, True                                      # ← chan

# ──────────────────────────────────────────────────────────────
#  Stage-2 – buggy signatures
# ──────────────────────────────────────────────────────────────
_JAVA_MODIFIERS = (
    "public", "protected", "private",
    "static", "final", "abstract", "native", "synchronized",
    "transient", "volatile", "strictfp",
)
_MOD_RX = re.compile(rf"^(?:{'|'.join(_JAVA_MODIFIERS)})\s+")

def _normalize(sig: str) -> str:
    sig = sig.strip("`").lstrip("-*# ").strip()

    # 1. strip Java modifiers (unchanged)
    while True:
        new = _MOD_RX.sub("", sig, count=1)
        if new == sig:
            break
        sig = new.strip()

    # 2. DROP THE RETURN TYPE  ← add this block
    parts = sig.split(None, 1)           # split on first whitespace
    if len(parts) == 2:                  # keep only the part AFTER it
        sig = parts[1]

    # 3. collapse internal whitespace (unchanged)
    return re.sub(r"\s+", " ", sig)

def check_rank_response(output: str, json_path: Path,
                        log_path: Optional[Path] = None) -> List[str]:
    lines = [ln.rstrip() for ln in output.splitlines()]
    idx = next((i for i, ln in enumerate(lines)
                if ln.lstrip("#").strip().lower().startswith("response:")),
               None)
    if idx is None:
        print("No 'RESPONSE:' line found in rank output.")
        _append_to_log("rank", [], log_path)
        return []

    raw = [ln.split(":", 1)[0].lstrip("-*# ").strip()
           for ln in lines[idx + 1:] if ln.strip()]

    buggy = {_normalize(sig)
             for cls in json.loads(json_path.read_text()).get("classes", [])
             for sig in cls.get("buggy_signatures", [])}

    seen, res = set(), []
    for r in raw:
        if _normalize(r) in buggy and r not in seen:
            seen.add(r)
            res.append(r)

    _append_to_log("rank", res, log_path)
    return res


# ──────────────────────────────────────────────────────────────
#  Stage-3 – extract "<signature>: line …" pairs    ← NEW
# ──────────────────────────────────────────────────────────────
def extract_line_response(output: str,
                          log_path: Optional[Path] = None) -> List[str]:
    """
    Grab every non-blank line that follows the last 'RESPONSE:' marker
    and contains a colon (':'), preserving order.

    Example expected lines:
        TimeSeries.getMinY():  return this.minY;
        TimeSeries.createCopy(...): line of code
    """
    lines = [ln.rstrip() for ln in output.splitlines()]
    idx = _find_response_marker(lines)
    if idx is None:
        print("No 'RESPONSE:' block found in source output.")
        _append_to_log("line", [], log_path)
        return []

    # collect lines after the marker (skip empty / code fences)
    pairs = [ln.strip()
             for ln in lines[idx + 1:]
             if ln.strip() and not ln.strip().startswith("```")]

    _append_to_log("line", pairs, log_path)
    return pairs
