import re
import subprocess
from pathlib import Path
import logging

LOGGER = logging.getLogger(__name__)

def read_lines(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]

def find_source_file(root: Path, rel: Path) -> Path:
    name = rel.name
    for path in root.rglob(name):
        if list(path.parts[-len(rel.parts):]) == list(rel.parts):
            return path
    LOGGER.critical("Failed to find source file for %s", rel)

def normalize_code(code: str) -> str:
    return re.sub(r' {3,}', '  ', code)