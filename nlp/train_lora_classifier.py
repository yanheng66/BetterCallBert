from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os

# ========== Set Up ==========
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DATA_PATH = "data/legal_classification_data.jsonl"
LABEL2ID = {"Contract": 0, "Tort": 1, "Criminal": 2, "Other": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)
OUTPUT_DIR = "legalbert-lora-finetuned"

import json
from datasets import Dataset

# ========== Step 1: Load JSONL ==========
with open(DATA_PATH, "r") as f:
    lines = [json.loads(line) for line in f if line.strip()]

dataset = Dataset.from_list(lines)

# ========== Step 2: Label Encoder ==========
def encode_label(example):
    example["label"] = LABEL2ID[example["label"]]
    return example

dataset = dataset.map(encode_label)

# ========== Step 3: Tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize_fn, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# ========== Step 4: LoRA Setup and Load ==========
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# ========== Step 5: Training Setup ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="no",
    save_strategy="epoch",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    logging_steps=10,
    save_total_limit=1,
    learning_rate=2e-5,
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)


# ========== Step 6: Start Training ==========
trainer.train()

# ========== Step 7: Save Model ==========
model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

print("✅ LoRA fine tune finished, save to:", os.path.join(OUTPUT_DIR, "final"))
