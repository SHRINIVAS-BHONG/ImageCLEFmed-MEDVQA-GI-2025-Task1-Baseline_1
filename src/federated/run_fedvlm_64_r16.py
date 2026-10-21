import argparse
import torch
import os
import sys
from typing import List, Tuple
from flwr.common import (
    FitRes, 
    FitIns, 
    Status, 
    Code, 
    ndarrays_to_parameters, 
    parameters_to_ndarrays
)

from .dataset import load_kvasir_vqa, partition_dataset
from .model_utils import (
    get_florence2_64_r16_model,
    extract_lora_b_parameters,
    extract_all_lora_parameters
)
from .plora_client import FedVLMPLoRAClient
from .plora_strategy import FedVLMStrategy

def main():
    parser = argparse.ArgumentParser(description="Run FedVLM & pLoRA on Florence-2 (r=16, batch=64)")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of hospital clients (default: 3)")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of federated communication rounds (default: 5)")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs per round (default: 1)")
    parser.add_argument("--partition_type", type=str, default="clinical_non_iid", 
                        choices=["clinical_non_iid", "dirichlet", "iid"], help="Hospital partitioning strategy")
    parser.add_argument("--baseline1_fedavg", action="store_true", help="If set, runs standard FedAvg instead of pLoRA")
    parser.add_argument("--dataset_coverage", type=float, default=1.0, help="Fraction of dataset to use (1.0 = full dataset)")
    parser.add_argument("--max_eval_samples", type=int, default=200, help="Number of samples for server evaluation")
    
    args = parser.parse_args()
    use_plora = not args.baseline1_fedavg
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n=================================================================")
    print("      FEDERATED MEDICAL VQA: FLORENCE-2 (r=16, batch=64)        ")
    print(f"  Method:          {'FedVLM & pLoRA (Baseline 3)' if use_plora else 'Parajuli FedAvg-LoRA (Baseline 1)'}")
    print(f"  Clients:         {args.num_clients} simulated hospitals")
    print(f"  Partition:       {args.partition_type}")
    print(f"  Rounds:          {args.num_rounds}")
    print(f"  Local Epochs:    {args.local_epochs}")
    print(f"  Device:          {device}")
    print("=================================================================\n")
    
    # 1. Load and partition dataset
    print("Loading SimulaMet-HOST/Kvasir-VQA dataset...")
    train_dataset, val_dataset = load_kvasir_vqa(dataset_coverage=args.dataset_coverage)
    client_train_subsets = partition_dataset(
        train_dataset, 
        num_clients=args.num_clients, 
        partition_type=args.partition_type
    )
    
    for i, subset in enumerate(client_train_subsets):
        print(f"  [Hospital {i}]: {len(subset)} training QA pairs")
        
    # 2. Initialize Shared Base Model & Processor
    print("\nInitializing Florence-2 (LoRA r=16, alpha=32)...")
    global_model, processor = get_florence2_64_r16_model(device=device)
    
    # Extract initial parameters
    if use_plora:
        initial_ndarrays, param_names = extract_lora_b_parameters(global_model)
        print(f"FedVLM pLoRA active: Synchronizing {len(param_names)} Matrix B weight matrices (Matrix A kept local).")
    else:
        initial_ndarrays, param_names = extract_all_lora_parameters(global_model)
        print(f"FedAvg active: Synchronizing all {len(param_names)} LoRA matrices.")
        
    current_parameters = ndarrays_to_parameters(initial_ndarrays)
    
    # 3. Initialize Clients
    print(f"\nInitializing {args.num_clients} hospital clients...")
    clients: List[FedVLMPLoRAClient] = []
    for client_idx in range(args.num_clients):
        client = FedVLMPLoRAClient(
            client_id=str(client_idx),
            model=global_model,
            processor=processor,
            train_subset=client_train_subsets[client_idx],
            val_subset=val_dataset.select(range(min(50, len(val_dataset)))),
            per_device_batch_size=4,
            gradient_accumulation_steps=16,
            base_learning_rate=9.6e-5,
            weight_decay=0.07,
            device=device,
            use_plora=use_plora
        )
        clients.append(client)

    # 4. Strategy Initialization
    strategy = FedVLMStrategy(
        eval_model=global_model,
        processor=processor,
        val_dataset=val_dataset,
        param_names=param_names,
        use_plora=use_plora,
        device=device,
        initial_parameters=current_parameters,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        max_eval_samples=args.max_eval_samples
    )

    # 5. Native Federated Communication Loop
    print("\n================ STARTING FEDERATED LEARNING ROUNDS ================")
    fit_config = {"local_epochs": str(args.local_epochs)}
    
    for server_round in range(1, args.num_rounds + 1):
        print(f"\n--- Communication Round {server_round}/{args.num_rounds} ---")
        client_results = []
        
        # Client local training
        for client_idx, client in enumerate(clients):
            print(f"  [Client {client_idx}] Training on local hospital partition...")
            current_ndarrays = parameters_to_ndarrays(current_parameters)
            updated_ndarrays, num_samples, metrics = client.fit(current_ndarrays, fit_config)
            
            fit_res = FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters(updated_ndarrays),
                num_examples=num_samples,
                metrics=metrics
            )
            # Tuple matching Flower strategy expectations: (client_proxy, fit_res)
            client_results.append((client, fit_res))
            print(f"  [Client {client_idx}] Training Loss: {metrics.get('train_loss', 0.0):.4f}")

        # Server-side Aggregation via Flower Strategy
        aggregated_parameters, _ = strategy.aggregate_fit(server_round, client_results, failures=[])
        if aggregated_parameters is not None:
            current_parameters = aggregated_parameters
            print(f"  [Server] Successfully aggregated Matrix B weights across all clients.")

        # Server-side Global Evaluation (ROUGE, BLEU, METEOR)
        print(f"  [Server] Evaluating global model on central validation set...")
        loss, eval_metrics = strategy.evaluate(server_round, current_parameters)
        
    print("\n================ FEDERATED TRAINING COMPLETED ================")
    print("All rounds finished successfully.")
    return current_parameters

if __name__ == "__main__":
    main()
