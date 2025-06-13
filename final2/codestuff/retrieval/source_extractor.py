import re, difflib
from pathlib import Path
from typing import List
from .utils import read_lines, find_source_file, normalize_code

def extract_method_by_sig(source: str, simple_sig: str) -> str:
    """
    Return the full source of the first method whose *header* matches
    `simple_sig`.  The match now tolerates an optional *throws* clause
    and any whitespace / line-breaks before the opening brace.
    """
    # 1️⃣  turn the user-supplied signature into a flexible regex
    esc = re.escape(simple_sig.strip())           # literal → escaped
    esc = esc.replace(r"\ ",  r"\s+")             # 1+ spaces/tabs
    esc = esc.replace(r"\t", r"\s+")              # (legacy)

    # allow optional "throws ..." before the '{'
    header_re = re.compile(
        esc + r"(?:\s*throws\b[^{]+)?\s*\{",
        re.MULTILINE | re.DOTALL
    )

    # 2️⃣  locate the header
    m = header_re.search(source)
    if not m:
        return f"// ‼ signature {simple_sig} not found"

    # 3️⃣  balance braces to capture the whole body
    start = m.start()
    idx   = m.end() - 1           # points at the '{'
    depth = 0
    while idx < len(source):
        if source[idx] == "{":
            depth += 1
        elif source[idx] == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
        idx += 1

    # Fallback (shouldn’t happen unless braces are unbalanced)
    return source[start:]



def remove_comments(code: str) -> str:
    """Remove // single‑line comments and /* … */ block comments."""
    comment_pattern = re.compile(r"/\*[\s\S]*?\*/|//.*?$", re.MULTILINE)
    return re.sub(comment_pattern, "", code)

MODIFIERS = (
    r"public|protected|private|static|abstract|native|"
    r"synchronized|strictfp"
)
SIG_START_RE = re.compile(rf"\b(?:{MODIFIERS})\b[^(]*\(")

def _clean_sig(sig: str) -> str:
    sig = re.sub(r"\s+", " ", sig).strip()
    cut = min([p for p in (sig.find("{"), sig.find(";")) if p != -1] or [len(sig)])
    return sig[:cut].strip()

def collect_signatures(code: str) -> List[str]:

    signatures: List[str] = []
    buffer: List[str] = []
    paren_depth = 0
    in_sig = False

    for raw in code.splitlines():
        line = raw.strip()
        if not line or line.startswith(("import ", "package ")):
            continue

        if not in_sig and SIG_START_RE.search(line):
            first_paren_idx = line.find("(")
            before_paren = line[:first_paren_idx]
            if "=" in before_paren:
                continue

            in_sig = True
            buffer = [line]
            paren_depth = line.count("(") - line.count(")")
            if paren_depth == 0:
                in_sig = False
                signatures.append(_clean_sig(" ".join(buffer)))
            continue

        if in_sig:
            buffer.append(line)
            paren_depth += line.count("(") - line.count(")")
            if paren_depth == 0:
                in_sig = False
                signatures.append(_clean_sig(" ".join(buffer)))

    return signatures


def extract_method(src: str, line_no: int) -> str:
    """From a 1‑based line number near the change, return the enclosing method."""
    lines = src.splitlines()
    idx = max(0, min(line_no - 1, len(lines) - 1))

    sig_re = re.compile(r"\b(public|protected|private|static|synchronized)\b")
    sig_idx = None
    for i in range(idx + 1, -1, -1):
        if sig_re.search(lines[i]) and "(" in lines[i]:
            sig_idx = i
            break
    if sig_idx is None:
        return ""

    brace_idx = None
    for j in range(sig_idx, len(lines)):
        if "{" in lines[j]:
            brace_idx = j
            break
    if brace_idx is None:
        return ""

    snippet = lines[sig_idx : brace_idx + 1]
    depth = lines[brace_idx].count("{") - lines[brace_idx].count("}")
    k = brace_idx + 1
    while k < len(lines) and depth > 0:
        snippet.append(lines[k])
        depth += lines[k].count("{") - lines[k].count("}")
        k += 1

    return "\n".join(snippet)


def gather_sources(buggy: Path, fixed: Path) -> list[dict]:
    mods = read_lines(buggy / "classes.modified")
    hunk_re = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    out: list[dict] = []

    for fqcn in mods:
        rel = Path(*fqcn.split(".")).with_suffix(".java")
        bsrc = find_source_file(buggy, rel)
        fsrc = find_source_file(fixed, rel)

        original_bcode = bsrc.read_text(encoding="utf8", errors="replace") if bsrc.exists() else ""
        original_fcode = fsrc.read_text(encoding="utf8", errors="replace") if fsrc.exists() else ""

        bcode = remove_comments(original_bcode)
        fcode = remove_comments(original_fcode)

        diff = list(
            difflib.unified_diff(
                original_bcode.splitlines(keepends=True),
                original_fcode.splitlines(keepends=True),
                fromfile=str(bsrc),
                tofile=str(fsrc),
                lineterm="",
            )
        )
        hunks, curr = [], []
        for ln in diff:
            if ln.startswith("@@ "):
                if curr:
                    hunks.append(curr)
                curr = [ln]
            elif curr:
                curr.append(ln)
        if curr:
            hunks.append(curr)

        methods: dict[str, dict] = {}
        for hunk in hunks:
            m = hunk_re.match(hunk[0])
            if not m:
                continue
            orig_start, new_start = int(m.group(1)), int(m.group(2))
            context = sum(1 for ln in hunk[1:] if ln.startswith(" "))
            bline, fline = orig_start + context - 1, new_start + context - 1

            mb = extract_method(original_bcode, bline)
            mf = extract_method(original_fcode, fline)
            sig = mb.splitlines()[0].strip() if mb else f"line_{bline}"

            entry = methods.setdefault(
                sig,
                {
                    "buggy_method": normalize_code(remove_comments(mb)),
                    "fixed_method": normalize_code(remove_comments(mf)),
                    "diff": [],
                    "changed_lines": 0,
                },
            )
            entry["diff"].extend(hunk)

        for e in methods.values():
            added = sum(1 for ln in e["diff"] if ln.startswith("+") and not ln.startswith("+++"))
            removed = sum(1 for ln in e["diff"] if ln.startswith("-") and not ln.startswith("---"))
            e["changed_lines"] = added + removed

        out.append(
            {
                "name": str(rel),
                "buggy_full_code": bcode,
                "fixed_full_code": fcode,
                "buggy_signatures": collect_signatures(bcode),
                "fixed_signatures": collect_signatures(fcode),
                "methods": list(methods.values()),
            }
        )

    return out
