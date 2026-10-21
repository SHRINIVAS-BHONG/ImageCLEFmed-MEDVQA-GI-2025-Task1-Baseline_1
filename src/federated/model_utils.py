import torch
import transformers.dynamic_module_utils
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, List, Tuple
import numpy as np

# 1. Bypass transformers checking for flash_attn (vital on Windows/standard PyTorch)
def enable_flash_attn_bypass():
    def custom_check_imports(filename):
        imports = transformers.dynamic_module_utils.get_imports(filename)
        missing_packages = []
        for imp in imports:
            if imp == "flash_attn":
                continue  # Ignore flash_attn
            try:
                import importlib
                importlib.import_module(imp)
            except ImportError:
                missing_packages.append(imp)

        if len(missing_packages) > 0:
            raise ImportError(
                f"Missing packages: {', '.join(missing_packages)}"
            )
        return transformers.dynamic_module_utils.get_relative_imports(filename)

    transformers.dynamic_module_utils.check_imports = custom_check_imports

def get_florence2_64_r16_model(
    model_id: str = "microsoft/Florence-2-base-ft",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05478,
    target_modules: List[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Initializes Florence-2 with LoRA rank 16 matching florence2_64_r16.ipynb.
    """
    enable_flash_attn_bypass()
    
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "lm_head", "fc2"]

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=lora_dropout,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model = model.to(device)
    
    return model, processor

def extract_lora_b_parameters(model: torch.nn.Module) -> Tuple[List[np.ndarray], List[str]]:
    """
    In FedVLM / pLoRA (Mitra et al.), only matrix B parameters are shared globally.
    Matrix A is kept strictly local to each hospital client.
    """
    param_list = []
    param_names = []
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param_list.append(param.detach().cpu().numpy())
            param_names.append(name)
    return param_list, param_names

def load_lora_b_parameters(model: torch.nn.Module, parameters: List[np.ndarray], param_names: List[str]):
    """
    Loads aggregated matrix B weights into the model while leaving personalized matrix A untouched.
    """
    state_dict = model.state_dict()
    for name, param_array in zip(param_names, parameters):
        if name in state_dict:
            state_dict[name].copy_(torch.tensor(param_array))
        else:
            raise KeyError(f"Parameter {name} not found in model state dict.")

def extract_all_lora_parameters(model: torch.nn.Module) -> Tuple[List[np.ndarray], List[str]]:
    """
    Standard FedAvg extraction (both A and B) for Baseline 1 comparison.
    """
    param_list = []
    param_names = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            param_list.append(param.detach().cpu().numpy())
            param_names.append(name)
    return param_list, param_names

def load_all_lora_parameters(model: torch.nn.Module, parameters: List[np.ndarray], param_names: List[str]):
    """
    Standard FedAvg loading (both A and B) for Baseline 1 comparison.
    """
    state_dict = model.state_dict()
    for name, param_array in zip(param_names, parameters):
        if name in state_dict:
            state_dict[name].copy_(torch.tensor(param_array))
