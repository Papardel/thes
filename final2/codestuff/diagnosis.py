import torch
from pathlib import Path
from typing import List, Union

def infer(phase: str,
                  prompt_files: Union[Path, List[Path]],
                  model,
                  tokenizer,
                  device: str,
                  max_length: int = 256):
    if isinstance(prompt_files, Path):
        prompt_files = [prompt_files]

    for prompt_file in prompt_files:
        text = prompt_file.read_text(encoding="utf-8")
        inputs = tokenizer(text, return_tensors="pt").to(device)

        output_ids = model.generate(**inputs, max_new_tokens=max_length, use_cache=False)
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        out_path = prompt_file.with_suffix(".out")
        out_path.write_text(result, encoding="utf-8")

        print(f"[{phase}] Inference result saved to {out_path}")
        return out_path