#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import logging

from .utils            import read_lines
from .source_extractor import gather_sources
from .failure_parser   import parse_failing_tests
from utils.datasets.d4j import Defects4JFetcher

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

def get_context(project: str, bug_id: str, out: str, flags: str = ""):
    fetcher = Defects4JFetcher(Path("."))
    buggy = fetcher.checkout(project, bug_id, "b")
    fixed = fetcher.checkout(project, bug_id, "f")

    for repo in (buggy, fixed):
        subprocess.run([fetcher.bin, "compile", "-w", str(repo)], check=True)

    subprocess.run([fetcher.bin, "test", "-w", str(buggy)], capture_output=True, check=True)

    fetcher.export_property(buggy, "tests.trigger", buggy / "tests.trigger")
    fetcher.export_property(buggy, "classes.modified", buggy / "classes.modified")

    failed_test_stack = read_lines(buggy / "failing_tests")
    classes = gather_sources(buggy, fixed)
    
    result = {
        "bug_id":       bug_id,
        "failed_tests": parse_failing_tests(failed_test_stack, buggy),
        "classes":      classes
    }

    out = Path(out + ".json")
    out.write_text(json.dumps(result, indent=2))
    LOGGER.info("Wrote %s", out)
    return buggy, fixed

if __name__ == "__main__":
    if len(sys.argv) == 2 and "-" in sys.argv[1]:
        proj, bid = sys.argv[1].split("-", 1)
    elif len(sys.argv) >= 3:
        proj, bid = sys.argv[1], sys.argv[2]
        out = sys.argv[3] if len(sys.argv) >= 4 else f"data/{proj}_{bid}_bug_info.json"
        flags = sys.argv[4] if len(sys.argv) >= 5 else ""
    else:
        sys.exit(f"Usage: {sys.argv[0]} <PROJECT> <BUG_ID> [flags]")

    get_context(proj, bid, "train full")