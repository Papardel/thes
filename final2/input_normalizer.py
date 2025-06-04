import json
from pathlib import Path

# STAGE 1 CHECKER AND PASSER
def extract_and_validate_response(output: str) -> str:
    """
    Search for the last occurrence of 'RESPONSE: ' in the output.
    """
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    prefix = "RESPONSE: "

    # find the *last* line that starts with 'RESPONSE: '
    idx = None
    for i in reversed(range(len(lines))):
        if lines[i].startswith(prefix):
            idx = i
            break

    if idx is None:
        raise ValueError(f"No line starts with '{prefix}' in the output.")

    fqcn = lines[idx][len(prefix):]
    if not fqcn:
        raise ValueError("No class name provided after 'RESPONSE: '.")
    return fqcn

def class_in_json(json_path: Path, fqcn: str) -> bool:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    target_path = fqcn.replace('.', '/') + '.java'
    print(f"Checking for class: {target_path!r} in JSON data.")
    for cls in data.get("classes", []):
        if cls.get("name") == target_path:
            return True
    return False

def check_class_response(output: str, json_path: Path) -> str:
    fqcn = extract_and_validate_response(output)
    if not class_in_json(json_path, fqcn):
        raise ValueError(f"Class '{fqcn}' not found in JSON classes list.")
    return fqcn


# STAGE 2 CHECKER AND PASSER

def check_rank_response(output: str, json_path: Path) -> list[str]:
    lines = [line for line in output.splitlines()]
    try:
        idx = next(i for i, line in enumerate(lines) if line.strip() == "RESPONSE:")
    except StopIteration:
        raise ValueError("No 'RESPONSE:' line found in output.")

    raw_sigs = [line.strip() for line in lines[idx + 1:] if line.strip()]
    if not raw_sigs:
        raise ValueError("No signatures found after 'RESPONSE:'.")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    buggy_set = set()
    for cls in data.get("classes", []):
        for sig in cls.get("buggy_signatures", []):
            buggy_set.add(sig)
    valid_sigs: list[str] = []
    for sig in raw_sigs:
        if sig in buggy_set:
            valid_sigs.append(sig)
        else:
            print(f"Signature not in JSON buggy_signatures: {sig!r}")

    return valid_sigs