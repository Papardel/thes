#!/usr/bin/env python3
"""
augment_phi3_outputs.py
=======================

Walk A_doneData/, find every   <Project>_<BugID>_bug_info_phi3_output.json
Get the matching              <Project>_<BugID>_bug_info.json   in A_unused_data/
Normalize each class path in that reference file and add

      "classes": ["org.foo.Bar", "org.foo.Baz"]

to the end of the phi-3 output object (overwrite in-place).

Usage
-----
python augment_phi3_outputs.py \
    --done-dir    A_doneData \
    --info-dir    A_unused_data
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List


def normalize(path: str) -> str:
    """org/jfree/data/XYZ.java → org.jfree.data.XYZ"""
    p = Path(path)
    p = ".".join(p.with_suffix("").parts)
    return p.rsplit(".", 1)[-1]



def extract_class_names(info_path: Path) -> List[str]:
    data = json.loads(info_path.read_text(encoding="utf-8"))
    classes = {normalize(c["name"]) for c in data.get("classes", []) if "name" in c}
    return sorted(classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--done-dir", required=True, help="Folder with *_phi3_output.json")
    ap.add_argument("--info-dir", required=True, help="Folder with *_bug_info.json")
    args = ap.parse_args()

    done_root = Path(args.done_dir).expanduser().resolve()
    info_root = Path(args.info_dir).expanduser().resolve()

    pattern = re.compile(r"^(.*_bug_info)_qwen_output\.json$")
    processed, skipped = 0, 0


    for phi3_file in done_root.glob("*_qwen_output.json"):
        m = pattern.match(phi3_file.name)
        if not m:
            skipped += 1
            continue

        info_file = info_root / f"{m.group(1)}.json"
        if not info_file.exists():
            print(f"[WARN] no matching info file for {phi3_file.name}")
            skipped += 1
            continue

        buggy_classes = extract_class_names(info_file)

        phi3_obj = json.loads(phi3_file.read_text(encoding="utf-8"))
        phi3_obj["classes"] = buggy_classes

        phi3_file.write_text(json.dumps(phi3_obj, indent=2), encoding="utf-8")
        processed += 1

    print(f"Updated {processed} files; skipped {skipped}.")


if __name__ == "__main__":
    main()
