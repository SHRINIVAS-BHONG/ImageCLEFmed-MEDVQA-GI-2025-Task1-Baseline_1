"""
CGGD-SlotLoc: Occupancy-Weighted Slot Attention for Auxiliary Counting & Location Supervision
Advancement over Florence-2 + LoRA Baseline for ImageCLEFmed-MEDVQA-GI-2025 (Kvasir-VQA)

Authors: Shrinivas Mukund Bhong et al.
"""

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

# ---------------------------------------------------------------------------
# Constants & Question Taxonomy (Exact match to baseline submission_task1_catwise.py)
# ---------------------------------------------------------------------------

QUESTION_TYPES = [
    "yes_no",
    "choice(single)",
    "choice(multiple)",
    "choice(color)",
    "location",
    "numerical_count",
]

COUNT_TYPE = "numerical_count"
LOCATION_TYPE = "location"
NUM_BINS = 7  # 0..5 exact count, 6 = '6+'
NUM_QUERIES = 16
PROJ_DIM = 768
LOCATION_MAX_CLASSES = 12
LOCATION_MIN_FREQ = 8

QUES2CATS = {
    'Have all polyps been removed?': 'yes_no',
    'Is this finding easy to detect?': 'yes_no',
    'Is there a green/black box artefact?': 'yes_no',
    'Is there text?': 'yes_no',
    'Does this image contain any finding?': 'yes_no',
    'What type of polyp is present?': 'choice(single)',
    'What type of procedure is the image taken from?': 'choice(single)',
    'What is the size of the polyp?': 'choice(single)',
    'Are there any abnormalities in the image? Check all that are present.': 'choice(multiple)',
    'Are there any anatomical landmarks in the image? Check all that are present.': 'choice(multiple)',
    'Are there any instruments in the image? Check all that are present.': 'choice(multiple)',
    'What color is the abnormality? If more than one separate with ;': 'choice(color)',
    'What color is the anatomical landmark? If more than one separate with ;': 'choice(color)',
    'Where in the image is the instrument?': 'location',
    'Where in the image is the abnormality?': 'location',
    'Where in the image is the anatomical landmark?': 'location',
    'How many findings are present?': 'numerical_count',
    'How many polyps are in the image?': 'numerical_count',
    'How many instrumnets are in the image?': 'numerical_count',  # baseline exact typo preserved
}

_unmapped_questions_seen = set()


def classify_question_type(question: str) -> str:
    """Classifies a question string into one of the 6 official Kvasir-VQA categories."""
    q = question.strip()
    if q in QUES2CATS:
        return QUES2CATS[q]
    _unmapped_questions_seen.add(q)
    return "unmapped"


def to_bin(n: int, num_bins: int = NUM_BINS) -> int:
    """Clamps a count integer to the maximum bin (e.g. 0..5, 6+)."""
    return min(max(n, 0), num_bins - 1)


def build_location_vocab(hf_dataset, min_freq: int = LOCATION_MIN_FREQ, max_classes: int = LOCATION_MAX_CLASSES) -> List[str]:
    """
    Builds closed-set location vocabulary from training dataset only (strict no-leakage).
    Answers with frequency < min_freq are excluded from closed-set head (labeled -100).
    """
    c = Counter()
    for rec in hf_dataset:
        if classify_question_type(rec["question"]) == LOCATION_TYPE:
            c[rec["answer"].strip().lower()] += 1
    vocab = [a for a, n in c.most_common(max_classes) if n >= min_freq]
    covered = sum(n for a, n in c.items() if a in vocab)
    total = sum(c.values())
    print(f"Location vocab ({len(vocab)} classes, min_freq={min_freq}, max_classes={max_classes}):")
    for a in vocab:
        print(f"    {a!r}  (n={c[a]})")
    print(f"Closed-set vocab covers {covered}/{total} ({covered / max(total, 1):.1%}) of location "
          f"answers in training set.")
    return vocab


def to_location_bin(answer: str, vocab: List[str]) -> int:
    """Maps a location answer string to vocabulary index or -100 if out of vocabulary."""
    a = answer.strip().lower()
    return vocab.index(a) if a in vocab else -100


# ---------------------------------------------------------------------------
# Dataset & Sampler
# ---------------------------------------------------------------------------

class KvasirVQADataset(Dataset):
    """
    Dataset wrapper adding question-type labels, count bin targets,
    and closed-set location targets to HuggingFace dataset examples.
    """
    def __init__(self, hf_dataset, location_vocab: List[str], name: str = "dataset"):
        self.ds = hf_dataset
        self.location_vocab = location_vocab
        self.qtypes = []
        self.count_bins = []
        self.location_bins = []
        type_counts = Counter()
        for rec in self.ds:
            q, a = rec["question"], rec["answer"]
            qtype = classify_question_type(q)
            count_bin = to_bin(int(a.strip())) if (qtype == COUNT_TYPE and re.fullmatch(r"\d+", a.strip())) else -100
            location_bin = to_location_bin(a, location_vocab) if qtype == LOCATION_TYPE else -100
            self.qtypes.append(qtype)
            self.count_bins.append(count_bin)
            self.location_bins.append(location_bin)
            type_counts[qtype] += 1

        n_loc_total = sum(1 for t in self.qtypes if t == LOCATION_TYPE)
        n_loc_covered = sum(1 for b in self.location_bins if b != -100)
        print(f"[{name}] {len(self.ds)} rows -- type distribution: {dict(type_counts)}")
        print(f"[{name}] location rows: {n_loc_total}, covered by closed-set vocab: {n_loc_covered} "
              f"({n_loc_covered / max(n_loc_total, 1):.1%})")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        rec = self.ds[idx]
        image = rec["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return {
            "image": image,
            "question": rec["question"],
            "answer": rec["answer"],
            "question_type": self.qtypes[idx],
            "count_bin": self.count_bins[idx],
            "location_bin": self.location_bins[idx],
            "img_id": rec.get("img_id", idx),
        }

    def type_distribution(self):
        return Counter(self.qtypes)

    def count_bin_distribution(self):
        return Counter(b for b in self.count_bins if b != -100)

    def location_bin_distribution(self):
        return Counter(b for b in self.location_bins if b != -100)

    def count_class_weights(self, num_bins: int = NUM_BINS, alpha: float = 0.180,
                            tau_floor: float = 0.335, tau_cap: float = 3.257) -> torch.Tensor:
        """Power-smoothed class weights to stabilize gradients on long-tail counting targets."""
        dist = self.count_bin_distribution()
        counts = torch.tensor([dist.get(b, 0) for b in range(num_bins)], dtype=torch.float)
        counts = torch.clamp(counts, min=1.0)
        med = torch.median(counts)
        w = torch.pow(med / counts, alpha)
        w = w / w.mean()
        w = torch.clamp(w, min=tau_floor, max=tau_cap)
        return w / w.mean()

    def location_class_weights(self, num_classes: int = LOCATION_MAX_CLASSES, alpha: float = 0.294,
                               tau_floor: float = 0.335, tau_cap: float = 3.257) -> Optional[torch.Tensor]:
        """Power-smoothed class weights for the closed-set location vocabulary."""
        if num_classes == 0:
            return None
        dist = self.location_bin_distribution()
        counts = torch.tensor([dist.get(b, 0) for b in range(num_classes)], dtype=torch.float)
        counts = torch.clamp(counts, min=1.0)
        med = torch.median(counts)
        w = torch.pow(med / counts, alpha)
        w = w / w.mean()
        w = torch.clamp(w, min=tau_floor, max=tau_cap)
        return w / w.mean()


def collate_fn(batch, processor):
    """Batches images, questions, answers, and auxiliary targets."""
    images = [b["image"] for b in batch]
    questions = [b["question"] for b in batch]
    answers = [b["answer"] for b in batch]
    qtypes = [b["question_type"] for b in batch]
    count_bins = torch.tensor([b["count_bin"] for b in batch], dtype=torch.long)
    location_bins = torch.tensor([b["location_bin"] for b in batch], dtype=torch.long)

    enc = processor(text=questions, images=images, return_tensors="pt", padding=True)
    with processor.tokenizer.as_target_tokenizer():
        labels = processor.tokenizer(answers, return_tensors="pt", padding=True).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "pixel_values": enc["pixel_values"],
        "labels": labels,
        "count_labels": count_bins,
        "location_labels": location_bins,
        "question_types": qtypes,
    }


class GuidedBatchSampler(Sampler):
    """
    Guarantees every batch has >=1 counting example AND (when batch_size >= 3)
    >=1 location example, preventing minority category neglect during gradient descent.
    """
    def __init__(self, dataset: KvasirVQADataset, batch_size: int, seed: int = 42):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.count_indices = [i for i, b in enumerate(dataset.count_bins) if b != -100]
        self.location_indices = [i for i, t in enumerate(dataset.qtypes) if t == LOCATION_TYPE]
        self.other_indices = [i for i, t in enumerate(dataset.qtypes) if t not in (COUNT_TYPE, LOCATION_TYPE)]
        self.n_batches = math.ceil(len(dataset) / batch_size)
        self.rng = random.Random(seed)

    def __iter__(self):
        count_pool = self.count_indices.copy() or list(range(1))
        if self.count_indices:
            self.rng.shuffle(count_pool)
        loc_pool = self.location_indices.copy()
        if loc_pool:
            self.rng.shuffle(loc_pool)
        other_pool = (self.other_indices.copy() if self.other_indices
                      else (self.count_indices + self.location_indices).copy())
        self.rng.shuffle(other_pool)
        cp = lp = op = 0

        for b in range(self.n_batches):
            batch = []
            guarantee_count_first = (b % 2 == 0) or not loc_pool
            has_dedicated_second_slot = self.batch_size >= 3 and bool(loc_pool) and bool(self.count_indices)

            if self.count_indices and (guarantee_count_first or not loc_pool):
                if cp >= len(count_pool):
                    self.rng.shuffle(count_pool)
                    cp = 0
                batch.append(count_pool[cp])
                cp += 1
            elif loc_pool:
                if lp >= len(loc_pool):
                    self.rng.shuffle(loc_pool)
                    lp = 0
                batch.append(loc_pool[lp])
                lp += 1

            if has_dedicated_second_slot:
                if guarantee_count_first:
                    if lp >= len(loc_pool):
                        self.rng.shuffle(loc_pool)
                        lp = 0
                    batch.append(loc_pool[lp])
                    lp += 1
                else:
                    if cp >= len(count_pool):
                        self.rng.shuffle(count_pool)
                        cp = 0
                    batch.append(count_pool[cp])
                    cp += 1

            while len(batch) < self.batch_size:
                if op >= len(other_pool):
                    self.rng.shuffle(other_pool)
                    op = 0
                batch.append(other_pool[op])
                op += 1

            self.rng.shuffle(batch)
            yield batch[:self.batch_size]

    def __len__(self):
        return self.n_batches


# ---------------------------------------------------------------------------
# Architecture Modules
# ---------------------------------------------------------------------------

class SlotAttentionResampler(nn.Module):
    """
    Perceiver-style slot attention resampler: NUM_QUERIES learned query vectors
    cross-attend to visual patch tokens over num_layers transformer blocks.
    """
    def __init__(self, dim: int = PROJ_DIM, num_queries: int = NUM_QUERIES, num_layers: int = 3, num_heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                cross_attn=nn.MultiheadAttention(dim, num_heads, batch_first=True),
                norm1=nn.LayerNorm(dim),
                ffn=nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
                norm2=nn.LayerNorm(dim),
            )) for _ in range(num_layers)
        ])

    def forward(self, patch_tokens: torch.Tensor, return_attn: bool = False):
        B = patch_tokens.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        last_attn = None
        for layer in self.layers:
            attn_out, attn_weights = layer["cross_attn"](
                q, patch_tokens, patch_tokens, need_weights=True, average_attn_weights=True
            )
            last_attn = attn_weights
            q = layer["norm1"](q + attn_out)
            q = layer["norm2"](q + layer["ffn"](q))
        if return_attn:
            return q, last_attn
        return q


def diversity_loss(queries: torch.Tensor, attn_weights: Optional[torch.Tensor] = None, spatial_weight: float = 0.5) -> torch.Tensor:
    """Encourages query slots to specialize by penalizing pairwise cosine and spatial overlap."""
    q = F.normalize(queries, dim=-1)
    sim = torch.bmm(q, q.transpose(1, 2))
    B, K, _ = sim.shape
    eye = torch.eye(K, device=sim.device, dtype=torch.bool)
    embed_term = sim.masked_select(~eye.unsqueeze(0).expand(B, -1, -1)).view(B, K, K - 1).mean()
    if attn_weights is None:
        return embed_term
    overlap = torch.bmm(attn_weights, attn_weights.transpose(1, 2))
    spatial_term = overlap.masked_select(~eye.unsqueeze(0).expand(B, -1, -1)).view(B, K, K - 1).mean()
    return (1.0 - spatial_weight) * embed_term + spatial_weight * spatial_term


def attention_concentration_loss(attn_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Entropy minimization on cross-attention maps to encourage crisp visual localization."""
    attn_weights = attn_weights.clamp_min(eps)
    N = attn_weights.size(-1)
    entropy = -(attn_weights * attn_weights.log()).sum(dim=-1) / math.log(N)
    return entropy.mean()


def count_regression_loss(pred_count: torch.Tensor, count_labels: torch.Tensor, max_count: int = NUM_BINS - 1) -> torch.Tensor:
    """Smooth L1 (Huber) regression loss for continuous count prediction."""
    valid = count_labels != -100
    if not valid.any():
        return torch.zeros((), device=pred_count.device)
    target = count_labels[valid].float().clamp(max=float(max_count))
    pred = pred_count[valid]
    return F.smooth_l1_loss(pred, target, beta=1.0)


def count_to_pseudo_logits(pred_count: torch.Tensor, num_bins: int = NUM_BINS, sigma: float = 0.75) -> torch.Tensor:
    """Converts continuous count prediction to pseudo-logits over discrete bins via RBF distance."""
    bins = torch.arange(num_bins, device=pred_count.device, dtype=pred_count.dtype)
    dist_sq = (pred_count.unsqueeze(-1) - bins.unsqueeze(0)) ** 2
    return -dist_sq / (2.0 * (sigma ** 2))


def soft_centroid(attn_weights: torch.Tensor, grid_side: int, has_global_token: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates expected (row, col) coordinates in [0, 1] for each slot."""
    spatial_attn = attn_weights[:, :, 1:] if has_global_token else attn_weights
    spatial_attn = spatial_attn / spatial_attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    rows = torch.arange(grid_side, device=attn_weights.device, dtype=attn_weights.dtype).repeat_interleave(grid_side)
    cols = torch.arange(grid_side, device=attn_weights.device, dtype=attn_weights.dtype).repeat(grid_side)
    denom = max(grid_side - 1, 1)
    row_exp = (spatial_attn * rows.view(1, 1, -1)).sum(-1) / denom
    col_exp = (spatial_attn * cols.view(1, 1, -1)).sum(-1) / denom
    return row_exp, col_exp


def compute_spatial_histogram(attn_weights: torch.Tensor, occ_probs: torch.Tensor,
                              grid_side: int, spatial_bins: int = 3, has_global_token: bool = False) -> torch.Tensor:
    """Decodes cross-attention weights into a 3x3 spatial soft histogram for multi-region location answers."""
    spatial_attn = attn_weights[:, :, 1:] if has_global_token else attn_weights
    B, K, _ = spatial_attn.shape
    spatial_attn = spatial_attn / spatial_attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    spatial_attn_2d = spatial_attn.view(B * K, 1, grid_side, grid_side)
    binned = F.adaptive_avg_pool2d(spatial_attn_2d, (spatial_bins, spatial_bins))
    binned = binned.view(B, K, spatial_bins * spatial_bins)
    binned = binned / binned.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    occ_w = occ_probs.unsqueeze(-1)
    occ_sum = occ_w.sum(dim=1).clamp_min(1e-6)
    spatial_hist = (occ_w * binned).sum(dim=1) / occ_sum
    return spatial_hist  # (B, 9)


class SlotReadout(nn.Module):
    """
    Per-slot occupancy prediction plus occupancy-weighted pooling:
    1. Soft cardinality: sum of slot occupancy probabilities.
    2. Pooled feature: occupancy-weighted average of slot features.
    3. 3x3 spatial histogram: occupancy-weighted spatial attention distribution.
    4. Soft centroid: (row, col) coordinates for visualization.
    """
    def __init__(self, dim: int = PROJ_DIM, spatial_bins: int = 3):
        super().__init__()
        self.occ_head = nn.Linear(dim, 1)
        self.spatial_bins = spatial_bins

    def forward(self, query_feats: torch.Tensor, attn_weights: torch.Tensor,
                grid_side: int, has_global_token: bool = False) -> Dict[str, torch.Tensor]:
        occ_logits = self.occ_head(query_feats).squeeze(-1)            # (B, K)
        occ_probs = torch.sigmoid(occ_logits)
        soft_cardinality = occ_probs.sum(dim=1)                         # (B,)

        weights = occ_probs.unsqueeze(-1)                               # (B, K, 1)
        pooled_feat = (weights * query_feats).sum(1) / weights.sum(1).clamp_min(1e-6)

        spatial_hist = compute_spatial_histogram(
            attn_weights, occ_probs, grid_side, spatial_bins=self.spatial_bins, has_global_token=has_global_token
        )
        row_exp, col_exp = soft_centroid(attn_weights, grid_side, has_global_token=has_global_token)
        occ_sum = occ_probs.sum(1).clamp_min(1e-6)
        centroid_row = (occ_probs * row_exp).sum(1) / occ_sum
        centroid_col = (occ_probs * col_exp).sum(1) / occ_sum
        centroid = torch.stack([centroid_row, centroid_col], dim=-1)

        return {
            "occ_logits": occ_logits,
            "occ_probs": occ_probs,
            "soft_cardinality": soft_cardinality,
            "pooled_feat": pooled_feat,
            "centroid": centroid,
            "spatial_hist": spatial_hist,
        }


class CountHead(nn.Module):
    """Single-neuron regression head using Smooth L1 loss."""
    def __init__(self, dim: int = PROJ_DIM):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(dim + 1, dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, pooled_feat: torch.Tensor, soft_cardinality: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pooled_feat, soft_cardinality.unsqueeze(-1)], dim=-1)
        return self.regressor(x).squeeze(-1)


class LocationHead(nn.Module):
    """Multi-region location classifier taking pooled feature and 3x3 spatial histogram."""
    def __init__(self, dim: int = PROJ_DIM, num_classes: int = LOCATION_MAX_CLASSES, spatial_bins: int = 3):
        super().__init__()
        in_dim = dim + spatial_bins * spatial_bins
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, num_classes)
        )

    def forward(self, pooled_feat: torch.Tensor, spatial_hist: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pooled_feat, spatial_hist], dim=-1)
        return self.classifier(x)


class TaskConditioner(nn.Module):
    """
    Content-gated task conditioner. Count questions receive a count projection embedding,
    location questions receive a location class embedding combined with the spatial histogram,
    and other questions receive a projection of the pooled slot features.
    """
    def __init__(self, dim: int = PROJ_DIM, num_location_classes: int = LOCATION_MAX_CLASSES, spatial_bins: int = 3):
        super().__init__()
        self.count_proj = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.location_class_embed = nn.Embedding(num_location_classes, dim)
        self.spatial_hist_proj = nn.Linear(spatial_bins * spatial_bins, dim)
        self.proj_count = nn.Linear(dim * 2, dim)
        self.proj_location = nn.Linear(dim * 3, dim)
        self.proj_plain = nn.Linear(dim, dim)

    def forward(self, pooled_feat: torch.Tensor, pred_count: torch.Tensor, location_logits: torch.Tensor,
                spatial_hist: torch.Tensor, is_count_mask: torch.Tensor, is_location_mask: torch.Tensor) -> torch.Tensor:
        count_embed = self.count_proj(pred_count.unsqueeze(-1))
        count_token = self.proj_count(torch.cat([pooled_feat, count_embed], dim=-1))

        loc_probs = F.softmax(location_logits, dim=-1)
        soft_loc_embed = loc_probs @ self.location_class_embed.weight
        spatial_embed = self.spatial_hist_proj(spatial_hist)
        location_token = self.proj_location(torch.cat([pooled_feat, soft_loc_embed, spatial_embed], dim=-1))

        plain_token = self.proj_plain(pooled_feat)

        cm = is_count_mask.view(-1, 1).to(pooled_feat.dtype)
        lm = is_location_mask.view(-1, 1).to(pooled_feat.dtype)
        other = (1.0 - cm) * (1.0 - lm)
        token = cm * count_token + lm * location_token + other * plain_token
        return token.unsqueeze(1)


def resolve_fused_count_answer(question_type: str, gen_text: str, count_logits: Optional[torch.Tensor],
                                use_fusion: bool = False, cls_conf_threshold: float = 0.85,
                                num_bins: int = NUM_BINS, pred_count: Optional[torch.Tensor] = None) -> Tuple[str, bool]:
    """Confidence-gated override for counting."""
    if not use_fusion or question_type != COUNT_TYPE:
        return gen_text, False
    if count_logits is not None:
        probs = F.softmax(count_logits, dim=-1)[0]
        conf, pred_bin = probs.max(dim=-1)
        conf, pred_bin = float(conf.item()), int(pred_bin.item())
        if conf >= cls_conf_threshold and pred_bin < num_bins - 1:
            return str(pred_bin), True
    elif pred_count is not None:
        val = float(pred_count[0].item()) if torch.is_tensor(pred_count) else float(pred_count)
        pred_bin = int(round(val))
        pred_bin = min(max(pred_bin, 0), num_bins - 1)
        conf = max(0.0, 1.0 - 2.0 * abs(val - round(val)))
        if conf >= cls_conf_threshold and pred_bin < num_bins - 1:
            return str(pred_bin), True
    return gen_text, False


def resolve_fused_location_answer(question_type: str, gen_text: str, location_logits: Optional[torch.Tensor],
                                   location_vocab: List[str], use_fusion: bool = False,
                                   cls_conf_threshold: float = 0.85) -> Tuple[str, bool]:
    """Confidence-gated override for location."""
    if not use_fusion or question_type != LOCATION_TYPE or location_logits is None or not location_vocab:
        return gen_text, False
    probs = F.softmax(location_logits, dim=-1)[0]
    conf, pred_idx = probs.max(dim=-1)
    conf, pred_idx = float(conf.item()), int(pred_idx.item())
    if conf >= cls_conf_threshold:
        return location_vocab[pred_idx], True
    return gen_text, False


class CGGDFlorence2(nn.Module):
    """
    CGGD-SlotLoc Florence-2 Wrapper.
    Extends base Florence-2 with:
    - SlotAttentionResampler (16 queries cross-attending to vision tokens)
    - SlotReadout (occupancy-weighted pooling, 3x3 spatial histogram, centroid)
    - CountHead (Smooth L1 regression)
    - LocationHead (CE classification)
    - TaskConditioner (soft conditioning token prefix)
    """
    def __init__(self, base_model, location_vocab: List[str], use_resampler: bool = True,
                 use_diversity: bool = True, use_conditioning: bool = True,
                 use_attn_concentration: bool = True, spatial_diversity_weight: float = 0.5,
                 count_class_weights: Optional[torch.Tensor] = None,
                 location_class_weights: Optional[torch.Tensor] = None,
                 grid_side: Optional[int] = None, has_global_token: Optional[bool] = None):
        super().__init__()
        self.base = base_model
        self.location_vocab = location_vocab
        num_location_classes = max(len(location_vocab), 1)
        self.use_resampler = use_resampler
        self.use_diversity = use_diversity
        self.use_conditioning = use_conditioning
        self.use_attn_concentration = use_attn_concentration
        self.spatial_diversity_weight = spatial_diversity_weight
        self.count_class_weights = count_class_weights
        self.location_class_weights = location_class_weights
        self.grid_side = grid_side
        self.has_global_token = has_global_token

        self.resampler = SlotAttentionResampler()
        self.readout = SlotReadout()
        self.count_head = CountHead()
        self.location_head = LocationHead(num_classes=num_location_classes)
        self.conditioner = TaskConditioner(num_location_classes=num_location_classes)

        # Match base model dtype if base model is in half/bfloat16 precision
        if hasattr(base_model, "dtype") and base_model.dtype in (torch.bfloat16, torch.float16):
            self.resampler.to(base_model.dtype)
            self.readout.to(base_model.dtype)
            self.count_head.to(base_model.dtype)
            self.location_head.to(base_model.dtype)
            self.conditioner.to(base_model.dtype)

    def encode_image_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.base._encode_image(pixel_values)

    def _resolve_grid_side(self, num_patches: int) -> int:
        if self.grid_side is not None:
            return self.grid_side
        side = int(round(math.sqrt(num_patches)))
        if side * side == num_patches:
            self.grid_side = side
            self.has_global_token = False
            return side
        side_plus1 = int(round(math.sqrt(num_patches - 1)))
        if num_patches > 1 and side_plus1 * side_plus1 == num_patches - 1:
            self.grid_side = side_plus1
            self.has_global_token = True
            return side_plus1
        raise ValueError(
            f"Patch token count {num_patches} is neither a perfect square nor square+1."
        )

    def compute_embeds(self, pixel_values: torch.Tensor, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor, is_count_mask: Optional[torch.Tensor] = None,
                       is_location_mask: Optional[torch.Tensor] = None,
                       override_count_bin: Optional[int] = None,
                       override_location_class: Optional[int] = None):
        image_tokens = self.encode_image_tokens(pixel_values)
        B = image_tokens.size(0)
        extras = {
            "pred_count": None, "count_logits": None, "location_logits": None,
            "centroid": None, "spatial_hist": None, "soft_cardinality": None,
            "attn_weights": None, "div_loss": torch.zeros((), device=image_tokens.device),
            "conc_loss": torch.zeros((), device=image_tokens.device)
        }
        extra_embeds = None

        if self.use_resampler:
            query_feats, attn_weights = self.resampler(image_tokens, return_attn=True)
            grid_side = self._resolve_grid_side(image_tokens.size(1))
            slots = self.readout(query_feats, attn_weights, grid_side, has_global_token=self.has_global_token)
            pred_count = self.count_head(slots["pooled_feat"], slots["soft_cardinality"])
            location_logits = self.location_head(slots["pooled_feat"], slots["spatial_hist"])
            count_logits = count_to_pseudo_logits(pred_count, NUM_BINS)

            pred_count_for_cond = pred_count
            if override_count_bin is not None:
                forced_count = torch.full((B,), float(override_count_bin), device=pred_count.device, dtype=pred_count.dtype)
                pred_count_for_cond = forced_count
            location_logits_for_cond = location_logits
            if override_location_class is not None:
                forced = torch.full((B,), override_location_class, device=location_logits.device, dtype=torch.long)
                location_logits_for_cond = F.one_hot(forced, location_logits.size(-1)).float() * 20.0

            icm = is_count_mask if is_count_mask is not None else torch.zeros(B, device=image_tokens.device)
            ilm = is_location_mask if is_location_mask is not None else torch.zeros(B, device=image_tokens.device)
            if self.use_conditioning:
                extra_embeds = self.conditioner(slots["pooled_feat"], pred_count_for_cond,
                                                 location_logits_for_cond, slots["spatial_hist"], icm, ilm)
            if self.use_diversity:
                extras["div_loss"] = diversity_loss(query_feats, attn_weights, self.spatial_diversity_weight)
            if self.use_attn_concentration:
                extras["conc_loss"] = attention_concentration_loss(attn_weights)

            extras.update({
                "pred_count": pred_count, "count_logits": count_logits,
                "location_logits": location_logits, "centroid": slots["centroid"],
                "spatial_hist": slots["spatial_hist"], "soft_cardinality": slots["soft_cardinality"],
                "attn_weights": attn_weights,
            })

        text_embeds = self.base.get_input_embeddings()(input_ids)
        merged_image_embeds = image_tokens if extra_embeds is None else torch.cat([extra_embeds, image_tokens], dim=1)
        inputs_embeds = torch.cat([merged_image_embeds, text_embeds], dim=1)
        img_mask = torch.ones(inputs_embeds.size(0), merged_image_embeds.size(1),
                               device=inputs_embeds.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([img_mask, attention_mask], dim=1)
        return inputs_embeds, full_attention_mask, extras

    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None,
                count_labels: Optional[torch.Tensor] = None, location_labels: Optional[torch.Tensor] = None):
        is_count_mask = (count_labels != -100).float() if count_labels is not None else None
        is_location_mask = (location_labels != -100).float() if location_labels is not None else None
        inputs_embeds, full_attention_mask, extras = self.compute_embeds(
            pixel_values, input_ids, attention_mask, is_count_mask=is_count_mask, is_location_mask=is_location_mask
        )
        dev = inputs_embeds.device
        aux_count_loss = torch.zeros((), device=dev)
        aux_location_loss = torch.zeros((), device=dev)

        if extras["pred_count"] is not None and count_labels is not None:
            valid = count_labels != -100
            if valid.any():
                aux_count_loss = count_regression_loss(extras["pred_count"], count_labels)
        if extras["location_logits"] is not None and location_labels is not None:
            valid = location_labels != -100
            if valid.any():
                w = self.location_class_weights.to(dev) if self.location_class_weights is not None else None
                aux_location_loss = F.cross_entropy(extras["location_logits"], location_labels, weight=w, ignore_index=-100)

        outputs = self.base.language_model(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask, labels=labels)
        return outputs, aux_count_loss, extras["div_loss"], extras["conc_loss"], aux_location_loss

    @torch.no_grad()
    def generate_answer(self, input_ids: torch.Tensor, pixel_values: torch.Tensor,
                         attention_mask: torch.Tensor, is_count: Optional[Union[bool, torch.Tensor]] = None,
                         is_location: Optional[Union[bool, torch.Tensor]] = None,
                         override_count_bin: Optional[int] = None,
                         override_location_class: Optional[int] = None, **gen_kwargs):
        B = pixel_values.size(0)
        is_count_mask = None if is_count is None else (
            is_count if torch.is_tensor(is_count) else torch.tensor([float(is_count)] * B, device=pixel_values.device))
        is_location_mask = None if is_location is None else (
            is_location if torch.is_tensor(is_location) else torch.tensor([float(is_location)] * B, device=pixel_values.device))
        inputs_embeds, full_attention_mask, extras = self.compute_embeds(
            pixel_values, input_ids, attention_mask, is_count_mask=is_count_mask, is_location_mask=is_location_mask,
            override_count_bin=override_count_bin, override_location_class=override_location_class
        )
        gen_ids = self.base.language_model.generate(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask, **gen_kwargs)
        return gen_ids, extras

    def trainable_parameters(self):
        params = (list(self.resampler.parameters()) + list(self.readout.parameters()) +
                  list(self.count_head.parameters()) + list(self.location_head.parameters()) +
                  list(self.conditioner.parameters()))
        params += [p for p in self.base.parameters() if p.requires_grad]
        return params

    def trainable_parameter_groups(self):
        """Returns (auxiliary_parameters, lora_parameters) for split-LR optimization."""
        aux_params = (list(self.resampler.parameters()) + list(self.readout.parameters()) +
                      list(self.count_head.parameters()) + list(self.location_head.parameters()) +
                      list(self.conditioner.parameters()))
        lora_params = [p for p in self.base.parameters() if p.requires_grad]
        return aux_params, lora_params
