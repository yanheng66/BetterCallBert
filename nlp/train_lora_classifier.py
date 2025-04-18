import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType

# ✅ CUDA check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ CUDA Available:", torch.cuda.is_available())
print("💻 Using", device)

# ========== Route Setup ==========
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DATA_PATH = "data/legal_title_qa_dataset_5300.jsonl"
OUTPUT_DIR = "legalbert-title-lora"

# ========== Label Mapping ==========
with open(DATA_PATH, "r", encoding="utf-8") as f:
    titles = sorted({json.loads(line)["label"] for line in f if line.strip()})
LABEL2ID = {title: i for i, title in enumerate(titles)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# ========== Embeed ==========
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]
dataset = Dataset.from_list(data)

def encode_label(example):
    example["label"] = LABEL2ID[example["label"]]
    return example
dataset = dataset.map(encode_label)

# ========== Tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize_fn, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ========== Load Model and setup LoRA ==========
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config).to(device)

# ========== Training Setup ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="no",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    logging_steps=10,
    save_total_limit=1,
    remove_unused_columns=False,
    report_to=[]
)

# ========== Training ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)
trainer.train()

# ========== Save ==========
model = model.merge_and_unload()
model.config.label2id = LABEL2ID
model.config.id2label = ID2LABEL

model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

print("✅ Model Train Finished and Saved to:", os.path.join(OUTPUT_DIR, "final"))
