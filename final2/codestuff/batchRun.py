#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

DEFAULT_MODELS = ["QWEN", "DeepSeek"]

JSON_FILES = [
    "/home/victor/School/ThesisPipeV1/final2/new_data/Chart_1_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Chart_2_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Chart_3_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Cli_1_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Cli_2_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Cli_3_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Cli_4_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Cli_5_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Codec_1_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Codec_2_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Codec_3_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Codec_4_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Codec_5_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Collections_1_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Collections_2_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Collections_3_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Collections_4_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Collections_5_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Compress_1_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Compress_2_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Compress_3_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Compress_4_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Compress_5_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Csv_1_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Csv_2_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Csv_3_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Csv_4_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Gson_1_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JacksonCore_2_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JacksonCore_3_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JacksonCore_4_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JacksonCore_5_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JacksonDatabind_1_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JacksonDatabind_2_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JacksonDatabind_3_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JacksonDatabind_4_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JacksonDatabind_5_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JacksonXml_2_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JacksonXml_3_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JacksonXml_4_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JacksonXml_5_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Jsoup_1_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JxPath_1_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JxPath_3_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/JxPath_5_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Math_1_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Math_3_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Math_4_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Math_5_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Mockito_5_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Time_15_bug_info.json",
    "/home/victor/School/ThesisPipeV1/final2/new_data/Time_18_bug_info.json",
]

PIPELINE = Path(__file__).resolve().parent / "pipeline.py"

def run_pipeline(json_file: str, model: str) -> None:
    stem = Path(json_file).stem                    # e.g. "Chart_3_bug_info"
    if stem.endswith("_bug_info"):
        stem = stem[:-9]
    try:
        project, bug_id = stem.split("_", 1)
    except ValueError:
        print(f"⚠️  Cannot parse project/bug-id from {stem}")
        return

    out_dir = f"TESTS/{model.lower()}/{project}_{bug_id}"
    cmd = ["python", str(PIPELINE), model, project, bug_id, out_dir]

    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ──────────────────────────────────────────────────────────────
#  Main entry-point
# ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pipeline.py for each bug JSON with one or more models."
    )
    parser.add_argument(
        "-m", "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        metavar="MODEL",
        help="Model names in the order you want them executed "
             f"(default: {' '.join(DEFAULT_MODELS)})",
    )
    args = parser.parse_args()

    for model in args.models:                  # QWEN first, then DeepSeek
        for jf in JSON_FILES:
            try:
                run_pipeline(jf, model)
            except subprocess.CalledProcessError as exc:
                print(f"❌  Pipeline failed for {jf} [{model}]: {exc}")
                # continue to next combo rather than abort everything


if __name__ == "__main__":
    main()