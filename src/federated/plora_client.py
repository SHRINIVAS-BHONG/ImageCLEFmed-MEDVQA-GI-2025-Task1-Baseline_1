import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import flwr as flwr
import numpy as np
from typing import Dict, List, Tuple
from tqdm.auto import tqdm

from .model_utils import (
    extract_lora_b_parameters, 
    load_lora_b_parameters,
    extract_all_lora_parameters,
    load_all_lora_parameters
)
from .dataset import create_collate_fn

# Global registry to preserve client-specific Matrix A across Flower simulation rounds
CLIENT_A_REGISTRY: Dict[str, Dict[str, torch.Tensor]] = {}

class FedVLMPLoRAClient(flwr.client.NumPyClient):
    """
    Flower Client implementing FedVLM / pLoRA (Mitra et al., ECAI).
    - Preserves personalized Matrix A locally across rounds.
    - Synchronizes only Matrix B globally via Flower server aggregation.
    - Configured with effective batch size 64 (4 x 16), lr 9.6e-5, LoRA r=16.
    """
    def __init__(
        self,
        client_id: str,
        model: torch.nn.Module,
        processor,
        train_subset,
        val_subset,
        per_device_batch_size: int = 4,
        gradient_accumulation_steps: int = 16,
        base_learning_rate: float = 9.6e-5,
        weight_decay: float = 0.07,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_plora: bool = True  # If False, falls back to standard FedAvg (Baseline 1)
    ):
        self.client_id = str(client_id)
        self.model = model
        self.processor = processor
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.per_device_batch_size = per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.base_learning_rate = base_learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.use_plora = use_plora
        
        # Collate function matching original notebook
        self.collate_fn = create_collate_fn(processor)
        
        # Dataloaders
        self.train_loader = DataLoader(
            train_subset, 
            batch_size=per_device_batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn
        )
        self.val_loader = DataLoader(
            val_subset, 
            batch_size=per_device_batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn
        )
        
        # Cache parameter names
        if self.use_plora:
            _, self.param_names = extract_lora_b_parameters(self.model)
        else:
            _, self.param_names = extract_all_lora_parameters(self.model)
            
        # Initialize local Matrix A cache if first time
        if self.use_plora and self.client_id not in CLIENT_A_REGISTRY:
            self._save_local_a()

    def _save_local_a(self):
        """Saves current client's Matrix A tensors into persistent registry."""
        local_a = {}
        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                local_a[name] = param.detach().clone().cpu()
        CLIENT_A_REGISTRY[self.client_id] = local_a

    def _restore_local_a(self):
        """Restores client's personalized Matrix A tensors from registry."""
        if self.client_id in CLIENT_A_REGISTRY:
            local_a = CLIENT_A_REGISTRY[self.client_id]
            state_dict = self.model.state_dict()
            for name, tensor in local_a.items():
                if name in state_dict:
                    state_dict[name].copy_(tensor.to(self.device))

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        if self.use_plora:
            params, _ = extract_lora_b_parameters(self.model)
        else:
            params, _ = extract_all_lora_parameters(self.model)
        return params

    def set_parameters(self, parameters: List[np.ndarray]):
        if self.use_plora:
            # Load global Matrix B
            load_lora_b_parameters(self.model, parameters, self.param_names)
            # Restore persistent client-specific Matrix A
            self._restore_local_a()
        else:
            load_all_lora_parameters(self.model, parameters, self.param_names)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        # 1. Synchronize weights
        self.set_parameters(parameters)
        
        # 2. Setup optimizer with weight decay 0.07
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.base_learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # 3. Train locally
        epochs = int(config.get("local_epochs", 1))
        self.model.train()
        total_loss = 0.0
        step_count = 0
        
        optimizer.zero_grad()
        for epoch in range(epochs):
            pbar = tqdm(
                self.train_loader,
                desc=f"🏥 [Hospital {self.client_id}] Epoch {epoch+1}/{epochs}",
                total=len(self.train_loader),
                leave=True,
                dynamic_ncols=True
            )
            for step, batch in enumerate(pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()
                
                step_loss = loss.item() * self.gradient_accumulation_steps
                total_loss += step_loss
                step_count += 1
                running_loss = total_loss / step_count
                
                # Effective batch size = batch_size * gradient_accumulation_steps (4 * 16 = 64)
                if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Live dynamic progress bar update
                pbar.set_postfix({
                    "batch": f"{step+1}/{len(self.train_loader)}",
                    "loss": f"{step_loss:.4f}",
                    "avg_loss": f"{running_loss:.4f}",
                    "eff_batch": f"{(step+1)//self.gradient_accumulation_steps}/{max(1, len(self.train_loader)//self.gradient_accumulation_steps)}"
                })
                    
        # 4. Save updated local Matrix A
        if self.use_plora:
            self._save_local_a()
            
        avg_loss = total_loss / max(1, step_count)
        num_samples = len(self.train_subset)
        
        return self.get_parameters(config={}), num_samples, {"train_loss": avg_loss}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()
        
        total_eval_loss = 0.0
        eval_steps = 0
        
        eval_pbar = tqdm(
            self.val_loader, 
            desc=f"🧪 [Hospital {self.client_id}] Evaluating", 
            leave=False,
            dynamic_ncols=True
        )
        with torch.no_grad():
            for batch in eval_pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_eval_loss += outputs.loss.item()
                eval_steps += 1
                eval_pbar.set_postfix({"val_loss": f"{outputs.loss.item():.4f}"})
                
        avg_eval_loss = total_eval_loss / max(1, eval_steps)
        return avg_eval_loss, len(self.val_subset), {"val_loss": avg_eval_loss}
                
        avg_eval_loss = total_eval_loss / max(1, eval_steps)
        return avg_eval_loss, len(self.val_subset), {"val_loss": avg_eval_loss}
