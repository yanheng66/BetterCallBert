from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from sentence_transformers import SentenceTransformer, models, losses, InputExample
from torch.utils.data import DataLoader
import os
import json

# ========= Configuration =========
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DATA_PATH = "data/legal_classification_data.jsonl"
BATCH_SIZE = 8
EPOCHS = 10
OUTPUT_PATH = "./legalbert-embedding-lora"

# ========= Load Data =========
with open(DATA_PATH, "r") as f:
    raw_data = [json.loads(line) for line in f if line.strip()]

train_examples = [
    InputExample(texts=[item["text"], item["text"]]) for item in raw_data  # Using the same text to form positive pairs
]

# ========= Tokenizer & Model =========
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
word_model = models.Transformer(MODEL_NAME)

# ========= Add LoRA =========
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]  # Tunable modules in BERT (can also be ['key', 'query', 'value', 'dense'])
)

base_model = word_model.auto_model
base_model = get_peft_model(base_model, lora_config)
word_model.auto_model = base_model  # Replace internal structure of ST model

pooling_model = models.Pooling(word_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_model, pooling_model])

# ========= Dataset and Loss =========
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# ========= Training =========
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    output_path=OUTPUT_PATH,
    show_progress_bar=True
)

print("✅ LoRA fine-tuned embedding model completed, saved to:", OUTPUT_PATH)
# ✅ Finally add this line
# ✅ Manually save Transformer module
transformer_save_path = os.path.join(OUTPUT_PATH, "0_Transformer")
word_model.tokenizer.save_pretrained(transformer_save_path)
word_model.auto_model.save_pretrained(transformer_save_path)

# ✅ Then save the entire SentenceTransformer model
model.save(OUTPUT_PATH)