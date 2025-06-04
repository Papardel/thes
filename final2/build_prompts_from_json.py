#!/usr/bin/env python3

import json
import os
import argparse
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from retrieval.source_extractor import extract_method_by_sig
from typing import List

def build_class_prompt(json_path, tokenizer, max_tokens):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    bug_id = data.get('bug_id', '')
    failed_tests = data.get('failed_tests', {})

    prompts = []
    buffer = 50

    for test_class, tests in failed_tests.items():
        
        prefix = (
            "IMPORTANT: You must reply _exactly_ in this form: \nRESPONSE: <fully-qualified-class-name>\n -and nothing else. No extra text, no newlines before/after, no explanations."
            f"Bug ID: {bug_id}\n"
            f"Test Class: {test_class}\n"
        )
        parts = []
        for test in tests:
            method    = test.get('methodName', '')
            error     = test.get('error', '')
            message   = test.get('message', '')
            fail_line = test.get('fail_line', '')
            source    = test.get('test_source', '')
            stack     = '\n'.join(test.get('stack', []))

            # Remove multi-line comments
            source = re.sub(r'/\*[\s\S]*?\*/', '', source)
            # Remove single-line comments
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
            + "\nReturn **only** the fully-qualified Java *source* class that contains the bug.\n"
            "Do **not** return:\n"
            "- the test class itself (e.g. com.fasterxml.jackson.databind.jsontype.ext.ExternalTypeIdWithEnum1328Test),\n"
            "- any class whose simple name starts or ends with Test, Tests, TestUtil, TestUtils, or TestHelper,\n"
            "- any test-utility class such as org.junit.Assert or org.junit.jupiter.api.Assertions.\n"
        )

        # count tokens
        tok_ids = tokenizer(combined, return_tensors='pt').input_ids[0]
        if tok_ids.size(0) <= max_tokens - buffer:
            prompts.append(combined)
        else:
            # too big: one prompt per test
            for section in parts:
                single = (
                    prefix
                    + section
                    + "\nReturn **only** the fully-qualified Java *source* class that contains the bug.\n"
                    "Do **not** return:\n"
                    "- the test class itself (e.g. com.fasterxml.jackson.databind.jsontype.ext.ExternalTypeIdWithEnum1328Test),\n"
                    "- any class whose simple name ends with Test, Tests, TestUtil, TestUtils, or TestHelper,\n"
                    "- any test-utility class such as org.junit.Assert or org.junit.jupiter.api.Assertions.\n"
                )
                prompts.append(single)

    return prompts

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
    
def build_method_prompt(
    json_path: str,
    tokenizer,
    ctx_limit: int,
    buffer: int = 50
) -> List[str]:
    """
    Build one-or-many “method-ranking” prompts.

    If the combined prompt would overflow `ctx_limit-buffer`, we split and
    emit one prompt per failing test.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    bug_id       = data["bug_id"]
    failed_tests = data["failed_tests"]
    classes      = {c["name"].replace("/", ".").rstrip(".java"): c["buggy_signatures"]
                    for c in data["classes"]}

    # --- fixed: header has trailing blank line so candidate block is kept ---
    header = (
        "IMPORTANT: You must reply _exactly_ in this form:\n"
        "RESPONSE:\n"
        "<signature-1>\n<signature-2>\n<signature-3>\n<signature-4>\n<signature-5>\n"
        "-and nothing else. No extra text, no newlines before/after, no explanations.\n\n"
    )

    sig_blocks = []
    for fqcn, sigs in classes.items():
        compact = "\n".join(f"    - {_strip_modifiers(s)}" for s in sigs)
        sig_blocks.append(f"### {fqcn}\n{compact}\n")
    sig_section = ("\nCandidate source classes and their method signatures "
                   "(modifiers removed):\n\n" + "\n".join(sig_blocks))

    instr = (
        "\nYour task:\n"
        "List the FIVE most suspicious method (or constructor)"
    )

    prompts: List[str] = []
    for tests in failed_tests.values():
        parts = []
        for t in tests:
            # strip comments to preserve tokens
            src = re.sub(r"/\*[\s\S]*?\*/|//.*", "", t["test_source"]).strip()
            parts.append(
                "--------------------\n"
                f"Source:\n{src}\nStack:\n" + "\n".join(t["stack"]) + "\n"
            )
        test_block = "\n".join(parts)

        prefix = f"Bug ID: {bug_id}\n\n"
        prompt = header + prefix + test_block + sig_section + instr

        ntok = tokenizer(prompt, return_tensors="pt").input_ids.size(1)
        if ntok <= ctx_limit - buffer:
            prompts.append(prompt)
        else:
            # one failing-test per prompt
            for p in parts:
                sub = header + prefix + p + sig_section + instr
                prompts.append(sub)

    return prompts

def build_method_source_prompt(
    json_path: str,
    top5: List[str],
    tokenizer,
    ctx_limit: int,
    buffer: int = 50,
) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    bug_id  = data["bug_id"]
    classes = {c["name"].replace("/", ".").rstrip(".java"): c
               for c in data["classes"]}

    # helper to build *one* prompt for a subset of methods -------------------
    def _make_prompt(pairs: List[tuple]) -> str:
        format = "IMPORTANT: You must reply _exactly_ in this form: \nRESPONSE: <fully-qualified-signature> \n -and nothing else. No extra text, no newlines before/after, no explanations."
        blocks = []
        for idx, (fq_sig, body) in enumerate(pairs, 1):
            blocks.append(
                f"### Method {idx}\n```java\n{body}\n```\n"
            )
        instr = (
            "Analyse the candidate methods above and determine the one most likely to cause the bug.\n"
        )
        return format + f"Bug ID: {bug_id}\n Chosen Methods\n\n" + \
               "\n".join(blocks) + instr

    # gather method bodies ---------------------------------------------------
    pairs: List[tuple] = []
    for fq_sig in top5:
        fqcn, simple = fq_sig.rsplit(".", 1)
        code = classes[fqcn]["buggy_full_code"]
        body = extract_method_by_sig(code, simple)
        pairs.append((fq_sig, body))

    # try big prompt first ---------------------------------------------------
    big = _make_prompt(pairs)
    ntok = tokenizer(big, return_tensors="pt").input_ids.size(1)
    if ntok <= ctx_limit - buffer:
        return [big]

    # fallback: 5 separate prompts, each trivially within limit --------------
    single_prompts = [_make_prompt([p]) for p in pairs]
    return single_prompts


def process_file(input_path, output_dir, tokenizer, model, device, max_length):
    """
    Generate responses for a single JSON file and write results to output_dir.
    """
    # get the model's real context window size
    context_size = tokenizer.model_max_length

    # build prompts, splitting out tests if needed
    prompts = build_class_prompt(input_path, tokenizer, context_size)

    responses = []
    for idx, prompt in enumerate(prompts, start=1):
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                use_cache=False  # disable cache to avoid custom cache errors
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        responses.append({
            'prompt_index': idx,
            'prompt': prompt,
            'response': text
        })

    # write output JSON
    base = os.path.splitext(os.path.basename(input_path))[0]
    output_name = f"{base}_output.json"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'bug_id': base, 'results': responses}, f, indent=2)
    print(f"Wrote results to {output_path}")


