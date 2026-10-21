import transformers
def custom_check_imports(filename):
    imports = transformers.dynamic_module_utils.get_relative_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports
transformers.dynamic_module_utils.check_imports = custom_check_imports

import os
import sys
import json
import time
import platform
import subprocess
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel, PeftConfig
from evaluate import load

# Add src to sys.path to import cggd_slotloc
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from cggd_slotloc import (
    CGGDFlorence2,
    QUES2CATS,
    QUESTION_TYPES,
    COUNT_TYPE,
    LOCATION_TYPE,
    classify_question_type,
    build_location_vocab,
    resolve_fused_count_answer,
    resolve_fused_location_answer,
)

bleu = load("bleu")
rouge = load("rouge")
meteor = load("meteor")

val_dataset = load_dataset("SimulaMet/Kvasir-VQA-test", split="validation")
predictions = []

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_mem():
    return torch.cuda.memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0

initial_mem = get_mem()

# ✏️✏️--------EDIT SECTION 1: SUBMISSION DETAILS and MODEL LOADING --------✏️✏️#

SUBMISSION_INFO = {
    "Participant_Names": "Shrinivas Mukund Bhong",
    "Affiliations": "AIDS Lab",
    "Contact_emails": ["shrinivasbhong@gmail.com"],
    "Team_Name": "CGGD-SlotLoc",
    "Country": "India",
    "Notes_to_organizers": '''
        Advancement on Florence-2 + LoRA baseline for ImageCLEFmed-MEDVQA-GI-2025.
        Introduces CGGD-SlotLoc: Occupancy-Weighted Slot Attention for Auxiliary
        Counting & Multi-region Spatial Supervision in GI Endoscopy VQA.
    '''
}

BASE_MODEL_ID = "microsoft/Florence-2-base-ft"
ADAPTER_MODEL_ID = str(SRC_DIR.parent / "florence2_64_r16_local")
CGGD_CHECKPOINT_PATH = str(SRC_DIR.parent / "checkpoints" / "cggd_slotloc_best.pt")

print(f"Loading Base Model: {BASE_MODEL_ID}")
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
).to(device)

# Load training dataset to derive the exact location vocabulary (no leakage)
train_source_dataset = load_dataset("SimulaMet-HOST/Kvasir-VQA")["raw"]
train_split = train_source_dataset.train_test_split(test_size=0.1, seed=7)["train"]
location_vocab = build_location_vocab(train_split)

# Check if pre-trained adapter exists to wrap base model
if os.path.exists(ADAPTER_MODEL_ID):
    print(f"Loading Peft adapter from {ADAPTER_MODEL_ID}")
    base_model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_ID)

# Wrap with CGGDFlorence2 architecture
model_cggd = CGGDFlorence2(base_model, location_vocab=location_vocab).to(device)

if os.path.exists(CGGD_CHECKPOINT_PATH):
    print(f"Loading CGGD-SlotLoc checkpoint from {CGGD_CHECKPOINT_PATH}")
    state = torch.load(CGGD_CHECKPOINT_PATH, map_location=device)
    model_cggd.load_state_dict(state, strict=False)
else:
    print(f"Notice: Checkpoint {CGGD_CHECKPOINT_PATH} not found; running with initialized/adapter weights.")

model_cggd.eval()

# 🏁----------------END SUBMISSION DETAILS and MODEL LOADING -----------------🏁#

start_time, post_model_mem = time.time(), get_mem()
model_mem_used = round(post_model_mem - initial_mem, 2)

for idx, ex in enumerate(tqdm(val_dataset, desc="Validating CGGD-SlotLoc")):
    question = ex["question"]
    image = ex["image"].convert("RGB") if ex["image"].mode != "RGB" else ex["image"]
    qtype = classify_question_type(question)

    inputs = processor(text=[question], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if k not in ['labels']}
    if "pixel_values" in inputs and torch.cuda.is_available():
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.no_grad():
        gen_ids, extras = model_cggd.generate_answer(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            is_count=(qtype == COUNT_TYPE),
            is_location=(qtype == LOCATION_TYPE),
        )

    raw_answer = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
    # Generative answer without fusion (fusion defaults to OFF as established in research diagnostics)
    answer = raw_answer

    assert isinstance(answer, str), f"Generated answer at index {idx} is not a string"
    predictions.append({
        "index": idx,
        "img_id": ex["img_id"],
        "question": ex["question"],
        "answer": answer,
        "question_category": qtype,
    })

assert len(predictions) == len(val_dataset), "Mismatch between predictions and dataset length"

total_time = round(time.time() - start_time, 4)
final_mem = round(get_mem() - post_model_mem, 2)

# Calculate overall NLG metrics
references = [[e] for e in val_dataset["answer"]]
preds = [pred["answer"] for pred in predictions]

bleu_result = bleu.compute(predictions=preds, references=references)
rouge_result = rouge.compute(predictions=preds, references=references)
meteor_result = meteor.compute(predictions=preds, references=references)
exact_match = sum(int(p.strip().lower() == r[0].strip().lower()) for p, r in zip(preds, references)) / len(preds)

public_scores = {
    "bleu": round(bleu_result["bleu"], 4),
    "rouge1": round(float(rouge_result["rouge1"]), 4),
    "rouge2": round(float(rouge_result["rouge2"]), 4),
    "rougeL": round(float(rouge_result["rougeL"]), 4),
    "meteor": round(float(meteor_result["meteor"]), 4),
    "exact_match": round(exact_match, 4),
}
print("✨ Overall Public scores:", public_scores)

# Category-wise score calculation
cat2preds = defaultdict(list)
cat2refs = defaultdict(list)
for i, pred in enumerate(predictions):
    cat = pred["question_category"]
    cat2preds[cat].append(pred["answer"])
    cat2refs[cat].append([val_dataset[i]["answer"]])

category_public_scores = dict()
for cat in cat2preds:
    b_res = bleu.compute(predictions=cat2preds[cat], references=cat2refs[cat])
    r_res = rouge.compute(predictions=cat2preds[cat], references=cat2refs[cat])
    m_res = meteor.compute(predictions=cat2preds[cat], references=cat2refs[cat])
    em_score = sum(int(p.strip().lower() == r[0].strip().lower()) for p, r in zip(cat2preds[cat], cat2refs[cat])) / len(cat2preds[cat])

    category_public_scores[cat] = {
        "n": len(cat2preds[cat]),
        "bleu": round(b_res["bleu"], 4),
        "rouge1": round(float(r_res["rouge1"]), 4),
        "rouge2": round(float(r_res["rouge2"]), 4),
        "rougeL": round(float(r_res["rougeL"]), 4),
        "meteor": round(float(m_res["meteor"]), 4),
        "exact_match": round(em_score, 4),
    }

output_dir = SRC_DIR.parent
with open(output_dir / "cat2preds_cggd.json", "w") as f:
    json.dump(cat2preds, f, indent=4)

with open(output_dir / "cat2refs_cggd.json", "w") as f:
    json.dump(cat2refs, f, indent=4)

print("\n📊 Metrics by Question Category Type:")
for cat, scores in category_public_scores.items():
    print(f"\n🔹 Category: {cat}")
    for metric, val in scores.items():
        print(f"  {metric}: {val}")

output_data = {
    "submission_info": SUBMISSION_INFO,
    "public_scores": public_scores,
    "category_public_scores": category_public_scores,
    "predictions": predictions,
    "total_time": total_time,
    "time_per_item": total_time / len(val_dataset),
    "memory_used_mb": final_mem,
    "model_memory_mb": model_mem_used,
    "gpu_name": gpu_name,
    "debug": {
        "packages": json.loads(subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"])),
        "system": {
            "python": platform.python_version(),
            "os": platform.system(),
            "platform": platform.platform(),
            "arch": platform.machine(),
        }
    }
}

with open(output_dir / "predictions_cggd.json", "w") as f:
    json.dump(output_data, f, indent=4)

print(f"\nTime: {total_time}s | Mem: {final_mem}MB | Model Load Mem: {model_mem_used}MB | GPU: {gpu_name}")
print("✅ Results successfully saved to 'predictions_cggd.json', 'cat2preds_cggd.json', and 'cat2refs_cggd.json'.")
