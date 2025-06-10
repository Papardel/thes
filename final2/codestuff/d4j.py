from pathlib import Path
import re
import shutil
import subprocess
from typing import List, Tuple
import logging

LOGGER = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

class Defects4JFetcher:

    def __init__(self, root: Path):
        self.root = root
        self.bin = shutil.which("defects4j")
        if not self.bin:
            raise RuntimeError("CLI NOT WORKING")
        self.root.mkdir(parents=True, exist_ok=True)

    def checkout(self, project: str, bug_id: str, version: str) -> Path:
        tgt = self.root / f"{project}_{bug_id}_{version}"
        if tgt.exists():
            return tgt
        subprocess.run(
            [self.bin, "checkout", "-p", project, "-v", f"{bug_id}{version}", "-w", str(tgt)],
            check=True, capture_output=True, text=True,
        )
        
        return tgt
    
    def test(self, project: str, bug_id: str, version: str):
        repo = self.checkout(project, bug_id, version)
        subprocess.run([self.bin, "test", "-w", str(repo)])

    def export_property(fetcher, root: Path, prop: str, out_file: Path) -> None:
        subprocess.run([
            fetcher.bin, "export",
            "-w", str(root),
            "-p", prop,
            "-o", str(out_file)
        ], check=True)