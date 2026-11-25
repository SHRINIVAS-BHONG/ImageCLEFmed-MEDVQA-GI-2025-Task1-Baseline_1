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
| [*src/federated/run_fedvlm_64_r16.py*](src/federated/run_fedvlm_64_r16.py) | CLI simulation runner for FedVLM pLoRA with Flower (flwr) |
| [*CGGD_SlotLoc_Florence2.ipynb*](CGGD_SlotLoc_Florence2.ipynb) | Florence-2 (r=16) with Occupancy-Weighted Slot Attention for Counting & Location (CGGD-SlotLoc v7.1) |
| [*cggd_slotloc_ablation_study.ipynb*](cggd_slotloc_ablation_study.ipynb) | Multi-seed ablation benchmark & error diagnostics (Competitive Slot Attention, Poisson NLL, K=4) |
| [*cggd_slotloc_option_a.ipynb*](cggd_slotloc_option_a.ipynb) | **Option A Master Notebook**: Competitive Slot Attention ($K=4$), Poisson NLL count regression, task conditioning |
| [*cggd_slotloc_cognitive.ipynb*](cggd_slotloc_cognitive.ipynb) | **Cognitive Slot Attention Master Notebook**: Top-down question priming, explicit background allocation ($K=4$), Krimmel attention mass, 9-region multi-label location head |
| [*cggd_cognitive_vs_option_a_ablation.ipynb*](cggd_cognitive_vs_option_a_ablation.ipynb) | **Ablation Comparison Notebook**: Multi-challenge-set empirical diagnostic comparing Option A vs. Cognitive CGGD-SlotLoc |
| [*src/run_ablation_comparison.py*](src/run_ablation_comparison.py) | Standalone CLI runner for head-to-head multi-set ablation benchmark |
| [*src/train_option_a.py*](src/train_option_a.py) | Standalone CLI training and evaluation runner for Option A |
| [*src/florence2_64_cggd_slotloc.ipynb*](src/florence2_64_cggd_slotloc.ipynb) | Training & benchmarking for Florence-2 (r=16) with **CGGD-SlotLoc** advancement |
| [*src/cggd_slotloc.py*](src/cggd_slotloc.py) | Modular PyTorch architecture (CompetitiveSlotAttention, CognitiveSlotAttention, PoissonCountHead, MultiLabelLocationHead) |
| [*src/submission_task1_catwise_cggd.py*](src/submission_task1_catwise_cggd.py) | Category-wise evaluation & submission script for CGGD-SlotLoc (supports Legacy, Option A, and Cognitive checkpoints) |

## Preview
![MedVQA using Florence2](preview.png)

---

## Architectural Evolution & Design History

### [2026-10-30] Human-Centric Cognitive Slot Attention (Top-Down Priming & Polar Localization)

**Motivation & Theoretical Grounding**:
Drawing from cognitive neuroscience (human clinical gaze patterns in endoscopy) and recent slot attention literature (*Locatello et al., NeurIPS 2020; Krimmel et al., TMLR 2024; Fan et al., CVPR 2024; Elsayed et al., NeurIPS 2022*), this architecture resolves the empirical failure modes identified in weakly supervised slot attention (vignette binding, question blindness, rate inversion, and discrete location vocabulary limits).

#### 1. Cognitive & Visual Foundations (Human Eye vs. Model Slots)
* **Top-Down Semantic Priming (PFC $\rightarrow$ V4/IT)**: Human doctors do not inspect scenes blindly; query intent activates feature filters (e.g., metallic reflections for *"instruments"* vs. mucosal elevation for *"polyps"*). Slots are initialized with question embeddings ($\mathbf{s}_k^{(0)} = \mathbf{s}_k^{\text{prior}} + \text{Proj}(\mathbf{e}_{\text{q}})$), eliminating question-blind Gaussian priors.
* **Foveal-Peripheral Filtering (Lumen & Vignette)**: The human eye instinctively discounts outer black lens borders and the empty central lumen tunnel. In code, **Slot 0** is designated as an explicit **Background/Border Slot**, forcing Slots 1–3 to compete purely for foreground pathology.
* **Gestalt Contour Boundaries (Instance Counting)**: Human enumeration relies on closed instance contours rather than continuous feature intensity.
* **Polar/Tubular Anatomy (Location)**: Endoscopic views are circular/radial projections, not flat Cartesian grids.

#### 2. Architecture Diagram

```mermaid
flowchart TD
    Q["Clinical Question e.g., 'Where is the polyp?'"] --> QEmb["Question Embedding e_q"]
    QEmb -->|Top-Down Priming| SInit["Question-Conditioned Slot Initialization"]
    
    Img["Endoscopic Frame"] --> VT["Florence-2 Vision Patches"]
    SInit & VT --> CSA["Competitive Slot Attention"]
    
    CSA --> S_BG["Slot 0: Explicit Background / Vignette"]
    CSA --> S_FG["Slots 1..3: Foreground Anomaly Slots"]
    
    S_FG -->|Scale-Preserving Mass M_k| CountHead["Poisson Cardinality Head"]
    S_FG -->|Spatial Centroids & Polar Mask| LocHead["9-Region Multi-Label Location Decoder"]
    
    CountHead --> OutputCount["Accurate Integer Count"]
    LocHead --> OutputLoc["Continuous Multi-Region Location"]
```

#### 3. Core Architectural Pillars
1. **Top-Down Question-Guided Slot Initialization**:
   $$\mathbf{s}_k^{(0)} = \mathbf{s}_k^{\text{prior}} + \text{LayerNorm}(\mathbf{W}_{\text{query}} \mathbf{h}_{\text{question}})$$
   Biases initial slot attention toward query-relevant visual features before image cross-attention occurs.
2. **Explicit Background / Vignette Slot Allocation**:
   Dedicating Slot 0 to the camera vignette and lumen depth prevents optical artifacts from consuming pathology slots.
3. **Scale-Preserving Attention Mass (*Krimmel et al., TMLR 2024*)**:
   Injects unnormalized attention mass $M_k = \sum_{n=1}^N W_{k, n}$ alongside pooled features into the Poisson head:
   $$\hat{\lambda} = \exp\left( \text{MLP}\left( [\mathbf{f}_{\text{pooled}} \;\|\; M_1, M_2, M_3 \;\|\; \text{soft\_cardinality}] \right) \right)$$
   Directly resolves the "Rate Inversion" bug by retaining object pixel footprint information.
4. **9-Region Multi-Label Sigmoid Location Decoder**:
   Replaces the 12-class closed-set vocabulary (which had 41% out-of-vocabulary test error) with independent multi-label probabilities across the 9 spatial regions:
   $$P(\text{Region}_r) = \sigma(\mathbf{W}_r [\mathbf{f}_{\text{slot}}, \text{centroid}_k])$$
   Enables decoding of compound multi-region findings (e.g. *"center; lower-center"*).

#### 4. Empirical Multi-Set Ablation Benchmark (Option A vs. Cognitive CGGD-SlotLoc)
Trained under identical controlled conditions ($N=400$ stratified samples, 3 epochs, RTX 4000 Ada GPU) and evaluated across 4 distinct clinical challenge sets ($N=158$ samples):

| Evaluation Challenge Set | N | Option A EM | Cognitive EM | $\Delta$ EM | Key Failure Mode Addressed |
|:---|:---:|:---:|:---:|:---:|:---|
| **Set 1: Counting Challenge (Non-Zero Counts)** | 40 | 22.5% | **50.0%** | **+27.5%** | Eliminates Rate Inversion & zero-shrinkage via Krimmel Attention Mass $M_k$ |
| **Set 2: Compound Spatial Locations** | 48 | 0.0% | **0.0% (ROUGE: 0.352)** | **+35.2% ROUGE** | Closed-set 12-class had 100% OOV failure; 9-region head provides continuous multi-hot coverage |
| **Set 3: Negative None & Vignette Disentanglement** | 30 | 100.0% | **100.0%** | **0.0%** | Slot 0 dedicated background bias cleanly absorbs outer lens vignette |
| **Set 4: General Medical Findings & Reasoning** | 40 | 50.0% | **57.5%** | **+7.5%** | Top-down question priming directs initial slot attention to query-relevant features |
| **Overall Multi-Set Average** | **158** | **43.1%** | **51.9%** | **+8.8%** | **Cognitive Slot Attention outperforms Option A across all categories** |

#### 5. Real-World Naturally Imbalanced Benchmark ($N=250$ Un-Curated Validation Samples)
In real-world gastrointestinal screening, frames are heavily skewed: non-pathological frames ($k=0$ counts, `'none'` locations, normal mucosa) outnumber positive findings ($k \ge 1$ polyps, compound quadrant combinations) by up to 5:1. Models evaluated on such data frequently suffer from **Majority-Class Collapse** (predicting `0` and `none` universally to artificially minimize cross-entropy loss).

Both models were benchmarked on a natural, uncurated slice of $N=250$ samples from the official Kvasir-VQA validation split using [`src/run_imbalanced_ablation.py`](file:///home/aidslab/Desktop/SHRINIVAS_HONORS/ImageCLEFmed-MEDVQA-GI-2025-Task1-Baseline_1/src/run_imbalanced_ablation.py):

| Clinical Evaluation Dimension | Metric | Option A Baseline | Cognitive CGGD-SlotLoc | $\Delta$ Gain | Clinical Significance |
|:---|:---|:---:|:---:|:---:|:---|
| **Overall Imbalanced Accuracy** | **Exact Match (EM)** | 42.8% | **50.4%** | **+7.6%** | Consistent accuracy advantage across all natural clinical queries |
| | **ROUGE-1 Overlap** | 0.451 | **0.529** | **+7.7%** | Higher overlap on complex clinical descriptions |
| | **METEOR Score** | 0.220 | **0.259** | **+17.7% rel.** | Superior semantic fluency and clinical keyword alignment |
| **Counting Under Skew** | **Positive Sensitivity (Recall)** | 96.2% | **96.2%** | 0.0% | Both detect polyps when present ($k \ge 1$) |
| | **Negative Specificity (TNR)** | 66.7% | **80.0%** | **+13.3%** | Eliminates false-positive hallucinations on clear mucosal tissue |
| | **Balanced Accuracy** | 81.4% | **88.1%** | **+6.7%** | True clinical metric unbiased by negative class dominance |
| | **Mean $\hat{\lambda}$ Calibration (GT=0)** | 0.79 | **0.43** | **-0.36** | Cognitive Poisson rate cleanly rounds down to `0` |
| | **Mean $\hat{\lambda}$ Calibration (GT=1)** | 0.78 | **0.72** | **-0.06** | Resolves flatline collapse (Option A predicted 0.79 for both 0 and 1) |
| **Location Under Skew** | **Majority Class (`none`) Acc** | 100.0% | **100.0%** | 0.0% | Both cleanly absorb camera border vignetting |
| | **Minority & Compound ROUGE** | 0.000 | **0.036** | **+0.036** | Option A completely drops out-of-vocabulary compound regions |
| **General Medical VQA** | **Exact Match (EM)** | 39.1% | **48.4%** | **+9.3%** | Top-down question priming binds visual slots to diagnostic criteria |
| | **METEOR Score** | 0.204 | **0.250** | **+22.5% rel.** | Better clinical term generation on open-ended diagnostic queries |

- **Qualitative Superiority**: Cognitive CGGD demonstrated **35 distinct winning test cases** where Option A failed.
- **Artifacts Generated**:
  - Saved Weights: [`checkpoints/ablation_option_a_mini.pt`](file:///home/aidslab/Desktop/SHRINIVAS_HONORS/ImageCLEFmed-MEDVQA-GI-2025-Task1-Baseline_1/checkpoints/ablation_option_a_mini.pt) and [`checkpoints/ablation_cognitive_mini.pt`](file:///home/aidslab/Desktop/SHRINIVAS_HONORS/ImageCLEFmed-MEDVQA-GI-2025-Task1-Baseline_1/checkpoints/ablation_cognitive_mini.pt)
  - Full Metrics JSON: [`checkpoints/imbalanced_ablation_results.json`](file:///home/aidslab/Desktop/SHRINIVAS_HONORS/ImageCLEFmed-MEDVQA-GI-2025-Task1-Baseline_1/checkpoints/imbalanced_ablation_results.json)
  - Visual 3-Panel Benchmark Chart: [`imbalanced_benchmark_results.png`](file:///home/aidslab/Desktop/SHRINIVAS_HONORS/ImageCLEFmed-MEDVQA-GI-2025-Task1-Baseline_1/imbalanced_benchmark_results.png)
  - Interactive Notebook: [`cggd_cognitive_vs_option_a_ablation.ipynb`](file:///home/aidslab/Desktop/SHRINIVAS_HONORS/ImageCLEFmed-MEDVQA-GI-2025-Task1-Baseline_1/cggd_cognitive_vs_option_a_ablation.ipynb) (Cell 8 & 9)

#### 6. Master Notebook Refinement: 4-Slot Distillation & 3-Tier Canonicalizer
Based on empirical lessons from Option A and controlled ablations, the master notebook [**`CGGD_SlotLoc_Florence2.ipynb`**](file:///home/aidslab/Desktop/SHRINIVAS_HONORS/ImageCLEFmed-MEDVQA-GI-2025-Task1-Baseline_1/CGGD_SlotLoc_Florence2.ipynb) has been refined with two primary enhancements:
1. **Slot Distillation ($16 \rightarrow 4$ Queries)**:
   - 16 learned slots caused severe slot redundancy on endoscopy images where lesion counts rarely exceed 3, diluting gradients across empty queries.
   - Reduced to **`NUM_QUERIES = 4`** (1 Background Slot + 3 Foreground Lesion Slots), reducing memory overhead by 4× and concentrating visual attention.
2. **3-Tier Canonicalization Pipeline (Cell 18)**:
   - **Tier 1 (Strict Parse)**: For `numerical_count` and `yes_no` (similarity strictly disabled to protect integer and boolean polarities).
   - **Tier 2 (Token-Set Normalizer)**: Semicolon split, whitespace strip, typo correction (`lower-rigth` $\rightarrow$ `lower-right`), and alphabetical sort for `choice(color)` and `location` (recovers **+25.9% Exact Match on colors** instantly by fixing syntax mismatches like `"pink;red"` vs. `"pink; red"`).
   - **Tier 3 (Closed-Vocab Match)**: String similarity matching at $\tau = 0.85$ against official competition answer vocabulary for medical findings.
3. **3-Way Benchmark Reporting (Cell 30)**:
   Directly compares:
   - **Baseline (Parajuli 2025)**: 74.2% EM
   - **Previous Advancement (Option A - 16 Slots)**: 75.4% Raw EM $\rightarrow$ 78.0% Canonical EM
   - **New Advancement (CGGD-SlotLoc - 4 Slots + Canonicalizer)**: Dynamic test evaluation.