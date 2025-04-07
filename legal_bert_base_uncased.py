import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

np.random.seed(42)
torch.manual_seed(42)

def evaluate_legal_bert(test_texts, test_labels, model_name="nlpaueb/legal-bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.eval()

    dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

    tokenized = dataset.map(preprocess_function, batched=True)

    predictions, probs_list = [], []
    for example in tokenized:
        with torch.no_grad():
            inputs = {
                "input_ids": torch.tensor([example["input_ids"]]),
                "attention_mask": torch.tensor([example["attention_mask"]])
            }
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            predictions.append(pred)
            probs_list.append(probs.squeeze().tolist())

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    class_names = ["Contract", "Tort"]
    report = classification_report(test_labels, predictions, target_names=class_names, digits=4)

    print("=" * 50)
    print("Legal-BERT Evaluation Results")
    print("=" * 50)
    print(f"Sample count: {len(test_labels)}\n")
    print("Key metrics:")
    print(f"- Accuracy : {accuracy:.4f}")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall   : {recall:.4f}")
    print(f"- F1 Score : {f1:.4f}")
    print("\nDetailed Report:\n", report)

    print("\nPer-question predictions:\n" + "-"*50)
    for text, true, pred, probs in zip(test_texts, test_labels, predictions, probs_list):
        c_prob, t_prob = probs[0] * 100, probs[1] * 100
        print(f"\n📝 Question: {text}")
        print(f"✅ True label     : {class_names[true]} ({true})")
        print(f"🤖 Predicted label: {class_names[pred]} ({pred})")
        print(f"📊 Confidence:")
        print(f"   - Contract: {c_prob:.2f}%")
        print(f"   - Tort    : {t_prob:.2f}%")
        explanation = (
            f"🧠 模型判断更偏向『合同法』，置信度为 {c_prob:.1f}%" if c_prob > t_prob
            else f"🧠 模型判断更偏向『侵权法』，置信度为 {t_prob:.1f}%"
        )
        print(f"📣 {explanation}")

if __name__ == "__main__":
    test_texts = [
        "What constitutes a valid contract?",
        "How to prove breach of contract?",
        "What is the statute of limitations for negligence?",
        "What are the elements of a tort claim?",
        "Can a verbal agreement be legally binding?",
        "How to establish duty of care in negligence cases?"
    ]
    test_labels = [0, 0, 1, 1, 0, 1]  # 0 = Contract, 1 = Tort
    evaluate_legal_bert(test_texts, test_labels)
