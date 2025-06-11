
# filepath: /home/victor/School/ThesisPipeV1/final2/modelLoader.py
import os
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAPTER_DIR = "training/adapters" 

class ModelLoader:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def loadModel(self, device: str | None = None, dtype: torch.dtype = torch.float16):
        selected = self.model_name.upper()
        if selected == "PHI":
            return self.load_phi3_base(device, dtype)
        elif selected == "DEEP":
            return self.load_deepseek_coder(device=device, dtype=dtype)
        elif selected == "QWEN":
            return self.load_qwen(device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def load_phi3_base(self, device: Optional[str] = None, dtype: torch.dtype = torch.float16):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        ).eval()
        return tok, model, device

    def load_qwen(self, *, device: Optional[str] = None, dtype: torch.dtype = torch.float16, full_gpu: bool = False):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if full_gpu or device != "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-Coder-7B-Instruct", torch_dtype=dtype, trust_remote_code=True
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True
            )
        model.eval()
        return tok, model, device

    def load_deepseek_coder(self, *, device: Optional[str] = None, dtype: torch.dtype = torch.float16):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        ).eval().to(device)
        return tok, model, device

    @staticmethod
    def attach_adapter(
        base_model,                       
        agent_id: int,                   
        adapter_name: str | None = None,
        trainable: bool = False,
    ):
        path = os.path.join(ADAPTER_DIR, f"agent{agent_id}")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Adapter folder not found: {path}")

        adapter_name = adapter_name or f"agent{agent_id}"

        if isinstance(base_model, PeftModel):
            if adapter_name in base_model.peft_config:
                base_model.set_adapter(adapter_name)           
            else:
                base_model.load_adapter(                        
                    path,
                    adapter_name=adapter_name,
                    is_trainable=trainable,
                )
                base_model.set_adapter(adapter_name)
            return base_model

        peft_model = PeftModel.from_pretrained(
            base_model,
            path,
            adapter_name=adapter_name,
            is_trainable=trainable,
        )
        peft_model.set_adapter(adapter_name)
        return peft_model           