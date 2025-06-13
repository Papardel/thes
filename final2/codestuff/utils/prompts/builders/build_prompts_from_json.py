#!/usr/bin/env python3

import json
import os
import argparse
from pathlib import Path
import random
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from retrieval.source_extractor import extract_method_by_sig
from typing import List
MODIFIERS = {
    "public", "protected", "private", "static",
    "abstract", "final", "native", "synchronized", "strictfp"
}

def _strip_modifiers(sig: str) -> str:
    """Remove leading Java modifiers to shorten a signature."""
    parts = sig.split()
    while parts and parts[0] in MODIFIERS:
        parts.pop(0)
    return " ".join(parts)

def _parse_method_signature(method_src: str) -> str:
    """Return the *declaration line* (no opening brace) of a Java method."""
    for line in method_src.splitlines():
        line = line.strip()
        if line:
            return line.rstrip("{").strip()
    raise ValueError("Empty method body provided")

SIG_RX = re.compile(r"[A-Za-z_]\w*\s*\(.*")

def trim_to_java_sig(text: str) -> str:
    m = SIG_RX.search(text.strip())
    if not m:
        raise ValueError(f"Cannot parse Java signature from: {text!r}")
    return m.group(0)

def default(classes) -> List[str]:
    """Original behaviour: first five `buggy_signatures` of the first class."""
    for cls in classes:
        sigs = cls.get("buggy_signatures") or []
        if sigs:
            fqcn = cls["name"].replace("/", ".").rstrip(".java")
            return [f"{fqcn}.{s}" for s in sigs[:5]]
    raise ValueError("No signatures found in JSON – cannot build source prompt")

def default_top5(json_path: Path) -> List[str]:
    """Pick one *buggy* method + four random others from `buggy_signatures`."""
    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    classes = data.get("classes", [])
    if not classes:
        raise ValueError("No classes found in JSON")

    primary_sig = None
    fqcn = None
    candidates = []
    for cls in classes:
        methods = cls.get("methods") or []
        for m in methods:
            try:
                simple_sig = _parse_method_signature(m["buggy_method"])
            except ValueError:
                continue
            if "(" in simple_sig and ")" in simple_sig:
                fqcn = cls["name"].replace("/", ".").rstrip(".java")
                primary_sig = f"{fqcn}.{simple_sig}"
                candidates = cls.get("buggy_signatures") or []
                break
        if primary_sig:
            break

    if primary_sig is None:
        return default(classes)

    def _simple(s: str) -> str:
        return s.split(None, 1)[-1]

    remaining = [s for s in candidates if s not in primary_sig]
    if len(remaining) < 4:
        pool = (remaining or candidates) * 5
    else:
        pool = remaining
    other_sigs = random.sample(pool, 4)

    return [primary_sig] + [f"{fqcn}.{s}" for s in other_sigs]

def build_class_prompt(json_path: Path, tokenizer, max_tokens: int) -> List[str]:
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    failed_tests = data.get('failed_tests', {})
    prompts = []
    buffer = 50

    for test_class, tests in failed_tests.items():
        prefix = f"Test Class: {test_class}\n"
        parts = []
        for test in tests:
            method    = test.get('methodName', '')
            error     = test.get('error', '')
            message   = test.get('message', '')
            fail_line = test.get('fail_line', '')
            source    = test.get('test_source', '')
            stack     = '\n'.join(test.get('stack', []))

            source = re.sub(r'/\*[\s\S]*?\*/', '', source)
            source = re.sub(r'//.*', '', source)
            source = source.strip()

            parts.append(
                f"Method: {method}\n"
                f"Error: {error}\n"
                f"Message: {message}\n"
                f"Failing Line: {fail_line}\n"
                f"Test Source:\n{source}\n"
                f"Stack Trace:\n{stack}\n"
            )

        combined = (
            prefix
            + '\n'.join(parts)
            + "\nIdentify the fully-qualified name of the non-test Java source classes that likely contain the bug causing the failure of the test cases above.\n"
            + "Analyze the test case sources, their stack traces, the error messages, and the failing lines to determine which classes are most likely responsible for the failure.\n"
            + "Pay attention to what class names are mentioned in the stack traces, error messages, and failing lines.\n"    
            + "\nReturn **only** the fully-qualified Java *source* class names.\n"
            + "Do **not** return:\n"
            + "- the test class itself (e.g. com.fasterxml.jackson.databind.jsontype.ext.ExternalTypeIdWithEnum1328Test),\n"
            + "- any class whose simple name starts or ends with Test, Tests, ClassTest, TestUtils, or TestHelper,\n"
            + "- any test-utility class such as org.junit.Assert or org.junit.jupiter.api.Assertions.\n"
            + "IMPORTANT: You must reply _exactly_ in this form:\n"
                    "RESPONSE:\n"
                    "<fully-qualified-class-name-1>\n"
                    "<fully-qualified-class-name-2>\n"
                    "<fully-qualified-class-name-3>\n"
                    "-and nothing else. No extra text, no newlines before/after, no explanations.\n\n"
        )

        tok_ids = tokenizer(combined, return_tensors='pt').input_ids[0]
        if tok_ids.size(0) <= max_tokens - buffer:
            prompts.append(combined)
        else:
            for section in parts:
                single = (
                    prefix
                    + section
                    + "\nReturn **only** the fully-qualified Java *source* class that contains the bug.\n"
                    + "Do **not** return:\n"
                    + "- the test class itself (e.g. com.fasterxml.jackson.databind.jsontype.ext.ExternalTypeIdWithEnum1328Test),\n"
                    + "- any class whose simple name starts or ends with Test, Tests, TestUtil, TestUtils, or TestHelper,\n"
                    + "- any test-utility class such as org.junit.Assert or org.junit.jupiter.api.Assertions.\n"
                    + "IMPORTANT: You must reply _exactly_ in this form:\n"
                    "RESPONSE:\n"
                    "<fully-qualified-class-name-1>\n"
                    "<fully-qualified-class-name-2>\n"
                    "<fully-qualified-class-name-3>\n"
                    "-and nothing else. No extra text, no newlines before/after, no explanations.\n\n"
                )
                prompts.append(single)

    return prompts

def build_method_prompt(
    json_path: Path,
    tokenizer,
    ctx_limit: int,
    buffer: int = 50
) -> List[str]:
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    failed_tests = data.get("failed_tests", {})
    classes      = {c["name"].replace("/", ".").rstrip(".java"): c["buggy_signatures"]
                    for c in data.get("classes", [])}

    sig_blocks = []
    for fqcn, sigs in classes.items():
        compact = "\n".join(f"    - {_strip_modifiers(s)}" for s in sigs)
        sig_blocks.append(f"### {fqcn}\n{compact}\n")
    sig_section = ("\nCandidate source classes and their method signatures "
                   "(modifiers removed):\n\n" + "\n".join(sig_blocks))

    instr = (
        "\nYour task:\n"
        "List the most likely methods that could cause the failure of the test cases above.\n"
        "Analyze the failing test(s) by looking at their code, the classes and methods they use, and the stack trace.\n"
        "IMPORTANT: Reply **ONLY** with the signatures, do not mention the class name. The arguments in the signatures must be the ones from the corresponding entry in the candidate list, **NOT** the source code. You **MUST** also return the type of the of the function, for example: **void MyClass** You must reply _exactly_ in this form:\n"
        "RESPONSE:\n"
        "<signature-1>\n"
        "<signature-2>\n"
        "<signature-3>\n"
        "<signature-4>\n"
        "<signature-5>\n"
        "-and nothing else. No extra text, no newlines before/after, no explanations.\n\n"
    )

    prompts: List[str] = []
    for tests in failed_tests.values():
        parts = []
        for t in tests:
            src = re.sub(r"/\*[\s\S]*?\*/|//.*", "", t.get("test_source", "")).strip()
            stack = "\n".join(t.get("stack", []))
            parts.append(f"Source:\n{src}\nStack:\n{stack}\n")
        test_block = "\n".join(parts)

        prompt = test_block + sig_section + instr

        ntok = tokenizer(prompt, return_tensors="pt").input_ids.size(1)
        if ntok <= ctx_limit - buffer:
            prompts.append(prompt)
        else:
            for p in parts:
                sub = p + sig_section + instr
                prompts.append(sub)

    return prompts

def build_method_source_prompt(
    json_path: Path,
    top5: List[str],
    tokenizer,
    ctx_limit: int,
    buffer: int = 50,
) -> List[str]:
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    classes = {c["name"].replace("/", ".").rstrip(".java"): c
               for c in data.get("classes", [])}

    test_snippets = []
    for test_list in data.get("failed_tests", {}).values():
        for t in test_list:
            src = re.sub(r"/\*[\s\S]*?\*/|//.*", "", t.get("test_source", "")).strip()
            stack_trace = "\n".join(t.get("stack", []))
            snippet = (
                "### Failing Test\n"
                "Test Source:\n```java\n" + src + "\n```\n"
                "Stack Trace:\n" + stack_trace + "\n"
            )
            test_snippets.append(snippet)

    test_block = "\n".join(test_snippets)

    def _make_prompt(method_pairs: List[tuple]) -> str:
        blocks = []
        for idx, (fq_sig, body) in enumerate(method_pairs, 1):
            blocks.append(
                f"### Candidate Method {idx}\n```java\n{body}\n```\n"
            )
        instructions = (
            "Analyse the failing test(s) above along with these candidate methods, these methods contain the causes of the test failures.\n"
            "Your task is to identify the most likely lines of code in these methods that could cause the failure of the test cases above.\n"
            "Look at the test cases, their stack traces, and the error messages to determine which lines in these methods are most likely responsible for the failure.\n"
            "Return the signature of the method and the lines of code that are most likely responsible for the failure.\n"
            "\nIMPORTANT: You must reply _exactly_ in this form:"
            "\nRESPONSE:\n<signature-1>: line of code\n line of code\n<signature-2>: line of code\n line of code\n<signature-3>: line of code\n line of code\n"
        )
        return (
            test_block
            + "\n"
            + "\n".join(blocks)
            + instructions
        )

    pairs: List[tuple] = []
    sorted_classes = sorted(classes.keys(), key=lambda k: -len(k))

    for sig in top5:
        sig = sig.strip()
        fqcn: str | None = None         
        body:  str | None = None         
        for cls_name in sorted_classes:
            if sig.startswith(cls_name + "."):
                fqcn        = cls_name
                method_part = trim_to_java_sig(sig[len(cls_name) + 1 :])
                code        = classes[fqcn]["buggy_full_code"]
                body        = extract_method_by_sig(code, method_part)
                break                                   

        if fqcn is None:
            m = re.search(r"(\w+)\s*\(", sig)
            if not m:
                print(f"Cannot parse method name from '{sig}'")
                continue
            method_name = m.group(1)

            for cls_name, cls_info in classes.items():
                try:
                    code = cls_info["buggy_full_code"]
                    body        = extract_method_by_sig(code, trim_to_java_sig(sig))

                    fqcn = cls_name
                    break
                except ValueError:
                    continue

        if fqcn is None or body is None:
            print(f" Method '{sig}' not found in any class.")
            continue

        pairs.append((f"{fqcn}.{sig}" if "." not in sig else sig, body))

    big_prompt = _make_prompt(pairs)
    return [big_prompt]

