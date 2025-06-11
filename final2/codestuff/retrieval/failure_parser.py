import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from .source_extractor import extract_method
from .utils import find_source_file, normalize_code

LOGGER = logging.getLogger(__name__)

STACK_RE = re.compile(
    r'^\s*at\s+([^\s]+)\.([^\s]+)\(([^:]+):(\d+)\)'
)

DEFAULT_IGNORE_PREFIXES: Tuple[str, ...] = (
    "at junit.",
    "at org.junit.",
    "at java.",
    "at jdk.",
    "at sun.",
    "at org.apache.tools.ant",
)

def parse_failing_tests(raw_lines: List[str], repo: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse Defects4J `failing_tests` output. If `repo` given,
    always attach the full test-source, and attach the failing line
    snippet only if a matching stack frame was found. Also summarize
    the stack trace into a concise list.

    Returns a dict mapping each test-class FQCN to its list of failures.
    """
    failures: List[Dict[str, Any]] = []
    i = 0
    n = len(raw_lines)

    while i < n:
        ln = raw_lines[i].strip()
        if not ln.startswith('--- '):
            i += 1
            continue
        header = ln[4:]
        if '::' in header:
            cls, mth = header.split('::', 1)
        else:
            parts = header.rsplit('.', 1)
            cls = parts[0]
            mth = parts[1] if len(parts) > 1 else ''

        # next line: error and message
        error = msg = ''
        if i + 1 < n:
            nxt = raw_lines[i + 1].strip()
            if ': ' in nxt:
                error, msg = nxt.split(': ', 1)
            else:
                error = nxt

        # collect stack frames until next '--- ' or EOF
        stack: List[str] = []
        j = i + 2
        while j < n and not raw_lines[j].startswith('--- '):
            if not raw_lines[j].startswith(DEFAULT_IGNORE_PREFIXES):
                stack.append(raw_lines[j])
            j += 1

        # build a concise summary of the stack trace
        summary_items: List[str] = []
        for frame in stack:
            m = STACK_RE.match(frame)
            if not m:
                continue
            fqcn_f, method_f, _, line_no_f = m.group(1), m.group(2), m.group(3), m.group(4)
            simple_cls = fqcn_f.split('.')[-1]
            summary_items.append(f"{simple_cls}.{method_f} line {line_no_f}")
        summary = ", ".join(summary_items)

        # find first matching frame for this test method
        fail_line: Optional[int] = None
        for frame in stack:
            m = STACK_RE.match(frame)
            if not m:
                continue
            fqcn_f, method_f, _, line_no_f = m.group(1), m.group(2), m.group(3), m.group(4)
            if fqcn_f == cls and method_f == mth:
                fail_line = int(line_no_f)
                break

        # base entry
        entry: Dict[str, Any] = {
            "methodName":  mth,
            "error":       error,
            "message":     msg,
            "fail_line":   "",
            "test_source": "",
            "stack":       [summary]
        }

        # attach snippet and test_source if repo is provided
        if repo is not None:
            rel = Path(*cls.split('.')).with_suffix('.java')
            test_file = find_source_file(repo, rel)
            if test_file and test_file.exists():
                src = test_file.read_text(encoding="utf8", errors="replace")
                lines = src.splitlines()

                # snippet: only if we got a concrete fail_line
                if fail_line is not None:
                    start = max(0, fail_line - 1)
                    end = min(len(lines), fail_line)
                    entry["fail_line"] = "\n".join(lines[start:end])

                # ALWAYS extract the full test method by locating its declaration
                method_re = re.compile(rf'^\s*public void {re.escape(mth)}\s*\(')
                decl_line: Optional[int] = None
                for idx, text in enumerate(lines, start=1):
                    if method_re.match(text):
                        decl_line = idx
                        break

                if decl_line is not None:
                    entry["test_source"] = normalize_code(extract_method(src, decl_line))
                else:
                    # fallback: extract around the first line of the file
                    entry["test_source"] = normalize_code(extract_method(src, 1))

        # stash the className for grouping, then drop it
        entry["_className"] = cls
        failures.append(entry)
        i = j

    # group by className
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for f in failures:
        cls = f.pop("_className")
        grouped.setdefault(cls, []).append(f)

    return grouped