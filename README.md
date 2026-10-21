# ImageClef2025MedVQA_Task1
This repository documents my submission for the ImageClef2025MedVQA Task1

## Source Description

| File | Description |
| -- | -- |
| [*preprint_draft_49217.pdf*](preprint_draft_49217.pdf) | Preprint |
| [*src/florence2_hyperparameter.ipynb*](src/florence2_hyperparameter.ipynb) | Code for optimal hyperparameter search |
| [*src/florence2_64_ft.ipynb*](src/florence2_64_ft.ipynb) | Training code for florence2 with vision tower frozen |
| [*src/florence2_64_ft_ef.ipynb*](src/florence2_64_ft_ef.ipynb) | Training code for florence2 with vision tower and encoder frozen |
| [*src/florence2_64_r8.ipynb*](src/florence2_64_r8.ipynb) | Training code for florence2 with LoRa rank 8 |
| [*src/florence2_64_r16.ipynb*](src/florence2_64_r16.ipynb) | Training code for florence2 with LoRa rank 16 (Baseline) |
| [*src/federated_florence2_64_r16.ipynb*](src/federated_florence2_64_r16.ipynb) | Federated Learning notebook using FedVLM & pLoRA on Florence-2 (r=16, batch=64) |
| [*src/federated/run_fedvlm_64_r16.py*](src/federated/run_fedvlm_64_r16.py) | CLI simulation runner for FedVLM pLoRA with Flower (flwr) |
| [*CGGD_SlotLoc_Florence2.ipynb*](CGGD_SlotLoc_Florence2.ipynb) | Florence-2 (r=16) with Occupancy-Weighted Slot Attention for Counting & Location (CGGD-SlotLoc v7.1) |
| [*src/florence2_64_cggd_slotloc.ipynb*](src/florence2_64_cggd_slotloc.ipynb) | Training & benchmarking for Florence-2 (r=16) with **CGGD-SlotLoc** advancement |
| [*src/cggd_slotloc.py*](src/cggd_slotloc.py) | Modular PyTorch architecture (SlotAttentionResampler, SlotReadout, CountHead, LocationHead, TaskConditioner) |
| [*src/submission_task1_catwise_cggd.py*](src/submission_task1_catwise_cggd.py) | Category-wise evaluation & submission script for CGGD-SlotLoc |

## Preview
![MedVQA using Florence2](preview.png)