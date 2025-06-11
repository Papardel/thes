import torch
from pathlib import Path
from typing import List, Union

def infer_base(phase: str,
                  prompt_files: Union[Path, List[Path]],
                  model,
                  tokenizer,
                  device: str,
                  max_length: int = 256,
                  modelT: str = "QWEN"):
        if modelT == "DEEP":
            return infer_deepseek(phase, prompt_files, model, tokenizer, device, max_length)
        elif modelT == "QWEN":
            return infer_qwen(phase, prompt_files, model, tokenizer, device, max_length)
        else:
            raise ValueError(f"Unsupported model type: {modelT}. Supported types are 'QWEN' and 'DEEP'.")
    
def QwenMsg(prt : str):
    return [{"role": "system", "content": "You are a helpful assistant, your job is to understand code"}, {"role": "user", "content": prt}]


def infer_qwen(phase: str,
                  prompt_files: Union[Path, List[Path]],
                  model,
                  tokenizer,
                  device: str,
                  max_length: int = 512):
    if isinstance(prompt_files, Path):
        prompt_files = [prompt_files]

    for prompt_file in prompt_files:
        text = prompt_file.read_text(encoding="utf-8")
        text = tokenizer.apply_chat_template(QwenMsg(text), tokenize = False, add_generation_prompt=True)

        inputs = tokenizer(text, return_tensors="pt").to(device)

        output_ids = model.generate(**inputs, max_new_tokens=max_length, use_cache=False)
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        out_path = prompt_file.with_suffix(".out")
        out_path.write_text(result, encoding="utf-8")

        print(f"[{phase}] Inference result saved to {out_path}")
        return out_path
    

def DeepMsg(prt : str):
    return [{ 'role': 'user', 'content': prt },]

def infer_deepseek(phase: str,
                  prompt_files: Union[Path, List[Path]],
                  model,
                  tokenizer,
                  device: str,
                  max_length: int = 512):
    if isinstance(prompt_files, Path):
        prompt_files = [prompt_files]

    for prompt_file in prompt_files:
        text = prompt_file.read_text(encoding="utf-8")
        tokenizer.pad_token       = tokenizer.eos_token      
        tokenizer.pad_token_id    = tokenizer.eos_token_id
        inputs = tokenizer.apply_chat_template(DeepMsg(text), add_generation_prompt=True, return_tensors="pt").to(model.device)
        output_ids = model.generate(inputs, max_new_tokens=max_length, use_cache=False, eos_token_id=tokenizer.eos_token_id)
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        out_path = prompt_file.with_suffix(".out")
        out_path.write_text(result, encoding="utf-8")

        print(f"[{phase}] Inference result saved to {out_path}")
        return out_path