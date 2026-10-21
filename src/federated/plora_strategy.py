import flwr as flwr
from flwr.common import Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import evaluate
import numpy as np

from .model_utils import (
    load_lora_b_parameters,
    load_all_lora_parameters,
    extract_lora_b_parameters,
    extract_all_lora_parameters
)
from .dataset import create_collate_fn

class FedVLMStrategy(flwr.server.strategy.FedAvg):
    """
    Custom Flower Strategy for FedVLM / pLoRA.
    Aggregates Matrix B updates from all hospital clients and evaluates
    on the global medical validation dataset with ROUGE, BLEU, and METEOR.
    """
    def __init__(
        self,
        eval_model: torch.nn.Module,
        processor,
        val_dataset,
        param_names: List[str],
        use_plora: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        eval_batch_size: int = 4,
        max_eval_samples: int = 200, # Subsample for fast validation during rounds if desired
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.eval_model = eval_model
        self.processor = processor
        self.val_dataset = val_dataset
        self.param_names = param_names
        self.use_plora = use_plora
        self.device = device
        self.eval_batch_size = eval_batch_size
        self.max_eval_samples = max_eval_samples
        
        # Load NLP metrics
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")
        
        # Validation loader
        collate_fn = create_collate_fn(processor)
        if max_eval_samples and max_eval_samples < len(val_dataset):
            eval_sub = val_dataset.select(range(max_eval_samples))
        else:
            eval_sub = val_dataset
            
        self.val_loader = DataLoader(
            eval_sub, 
            batch_size=eval_batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        self.eval_sub = eval_sub

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Global evaluation executed on the central server after every federated aggregation round.
        Computes standard CLEF MedVQA metrics: ROUGE, BLEU, METEOR.
        """
        if server_round == 0:
            return 0.0, {}

        # 1. Update evaluation model with aggregated weights
        ndarrays = parameters_to_ndarrays(parameters)
        if self.use_plora:
            load_lora_b_parameters(self.eval_model, ndarrays, self.param_names)
        else:
            load_all_lora_parameters(self.eval_model, ndarrays, self.param_names)

        self.eval_model.eval()
        predictions = []
        references = []
        total_eval_loss = 0.0
        step_count = 0

        # 2. Evaluate generation
        eval_pbar = tqdm(
            self.eval_sub, 
            desc=f"🌐 [Server Round {server_round}] Evaluating", 
            total=len(self.eval_sub),
            leave=True,
            dynamic_ncols=True
        )
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", dtype=torch.float16 if "cuda" in str(self.device) else torch.bfloat16):
                for idx, ex in enumerate(eval_pbar):
                    question = ex["question"]
                    image = ex["image"].convert("RGB") if ex["image"].mode != "RGB" else ex["image"]
                    ground_truth = str(ex["answer"]).strip()

                    inputs = self.processor(text=[question], images=[image], return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items() if k not in ['labels', 'attention_mask']}

                    output = self.eval_model.generate(**inputs, max_new_tokens=32)
                    predicted_text = self.processor.tokenizer.decode(output[0], skip_special_tokens=True).strip()

                    predictions.append(predicted_text)
                    references.append(ground_truth)
                    
                    eval_pbar.set_postfix({
                        "sample": f"{idx+1}/{len(self.eval_sub)}",
                        "pct": f"{(idx+1)/len(self.eval_sub)*100:.1f}%"
                    })

        # 3. Calculate metrics
        rouge_res = self.rouge.compute(predictions=predictions, references=references)
        bleu_res = self.bleu.compute(predictions=predictions, references=[[r] for r in references])
        meteor_res = self.meteor.compute(predictions=predictions, references=references)

        metrics = {
            "ROUGE-1": float(rouge_res.get("rouge1", 0.0)),
            "ROUGE-2": float(rouge_res.get("rouge2", 0.0)),
            "ROUGE-L": float(rouge_res.get("rougeL", 0.0)),
            "BLEU": float(bleu_res.get("bleu", 0.0)),
            "METEOR": float(meteor_res.get("meteor", 0.0)),
        }

        print(f"\n================ SERVER ROUND {server_round} METRICS ================")
        print(f"ROUGE-1: {metrics['ROUGE-1']:.4f} | ROUGE-2: {metrics['ROUGE-2']:.4f} | ROUGE-L: {metrics['ROUGE-L']:.4f}")
        print(f"BLEU:    {metrics['BLEU']:.4f} | METEOR:  {metrics['METEOR']:.4f}")
        print("===================================================================\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Loss placeholder (using negative ROUGE-L as optimization criterion)
        pseudo_loss = 1.0 - metrics["ROUGE-L"]
        return pseudo_loss, metrics
