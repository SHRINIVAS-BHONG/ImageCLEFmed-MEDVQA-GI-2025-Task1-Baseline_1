import numpy as np
from datasets import load_dataset, Dataset
import torch

def load_kvasir_vqa(seed: int = 7, test_size: float = 0.1, dataset_coverage: float = 1.0):
    """
    Loads Kvasir-VQA and partitions into train and global validation sets
    matching the original baseline notebook (florence2_64_r16.ipynb).
    """
    ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")['raw']
    
    if dataset_coverage < 1.0:
        sub = ds.train_test_split(test_size=dataset_coverage, seed=seed, shuffle=True)
        split_ds = sub['test'].train_test_split(test_size=test_size, seed=seed)
        train_dataset, val_dataset = split_ds['train'], split_ds['test']
    else:
        split_ds = ds.train_test_split(test_size=test_size, seed=seed)
        train_dataset, val_dataset = split_ds['train'], split_ds['test']
        
    return train_dataset, val_dataset

def partition_dataset(
    train_dataset: Dataset, 
    num_clients: int = 3, 
    partition_type: str = "clinical_non_iid", 
    seed: int = 7
):
    """
    Partitions the training dataset across simulated hospital clients.
    
    Partition types:
    1. 'clinical_non_iid': Simulates real hospital specialization
       - Client 0 (Colorectal Screening Unit): Polyps & Normal
       - Client 1 (Upper GI Clinic): Esophagitis & Normal
       - Client 2 (IBD & Therapeutic Unit): Ulcerative Colitis & Instruments
    2. 'dirichlet': Dirichlet non-IID partition (alpha=0.5)
    3. 'iid': Uniform random split across clients
    """
    np.random.seed(seed)
    total_samples = len(train_dataset)
    
    # Check if category column exists
    features = train_dataset.column_names
    cat_col = None
    for candidate in ['source', 'category', 'label', 'finding', 'organ']:
        if candidate in features:
            cat_col = candidate
            break
            
    if partition_type == "clinical_non_iid" and cat_col is not None:
        client_indices = [[] for _ in range(num_clients)]
        
        # Categorize indices
        categories = np.array(train_dataset[cat_col])
        unique_cats = np.unique(categories)
        
        # Hospital mapping:
        # Client 0: Polyps, Normal
        # Client 1: Esophagitis, Normal
        # Client 2: Ulcerative Colitis, Instruments
        for idx, cat in enumerate(categories):
            cat_str = str(cat).lower()
            if "polyp" in cat_str:
                client_indices[0].append(idx)
            elif "esophagitis" in cat_str or "gerd" in cat_str or "z-line" in cat_str:
                client_indices[1 % num_clients].append(idx)
            elif "colitis" in cat_str or "instrument" in cat_str:
                client_indices[2 % num_clients].append(idx)
            else: # Normal or shared
                # Distribute normal across all clients with realistic weights
                chosen_client = np.random.choice(num_clients, p=[0.4, 0.4, 0.2] if num_clients == 3 else None)
                client_indices[chosen_client].append(idx)
                
        # If any client is empty or too small, fall back to stratified split
        if any(len(c) == 0 for c in client_indices):
            return partition_dataset(train_dataset, num_clients, partition_type="dirichlet", seed=seed)
            
        client_subsets = [train_dataset.select(indices) for indices in client_indices]
        return client_subsets
        
    elif partition_type == "dirichlet" and cat_col is not None:
        alpha = 0.5
        categories = np.array(train_dataset[cat_col])
        unique_cats = np.unique(categories)
        client_indices = [[] for _ in range(num_clients)]
        
        for cat in unique_cats:
            cat_indices = np.where(categories == cat)[0]
            np.random.shuffle(cat_indices)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(cat_indices)).astype(int)[:-1]
            split_indices = np.split(cat_indices, proportions)
            for c_id, s_idx in enumerate(split_indices):
                client_indices[c_id].extend(s_idx)
                
        client_subsets = [train_dataset.select(indices) for indices in client_indices]
        return client_subsets
        
    else: # Default IID
        shuffled_indices = np.random.permutation(total_samples)
        client_splits = np.array_split(shuffled_indices, num_clients)
        return [train_dataset.select(indices.tolist()) for indices in client_splits]

def create_collate_fn(processor):
    """
    Creates the collation function matching florence2_64_r16.ipynb.
    """
    def collate_fn(batch):
        questions = [f"{x['question']}" for x in batch]
        images = [
            x["image"].convert("RGB") if x["image"].mode != "RGB" else x["image"] 
            for x in batch
        ]
        answers = [x["answer"] for x in batch]

        inputs = processor(text=questions, images=images, return_tensors="pt", padding=True)
        labels = processor.tokenizer(answers, return_tensors="pt", padding=True).input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    return collate_fn
