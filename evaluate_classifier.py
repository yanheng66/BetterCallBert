import json
from nlp.classifier import LegalClassifier

TEST_DATA_PATH = "data/classifier_test_data.jsonl"
with open(TEST_DATA_PATH, "r") as f:
    test_data = [json.loads(line) for line in f if line.strip()]

classifier = LegalClassifier(
    model_name="legalbert-lora-finetuned/final",
    label_map={0: "Contract", 1: "Tort", 2: "Criminal", 3: "Other"}
)

correct = 0
results = []

for item in test_data:
    question = item["text"]
    true_label = item["label"]
    result = classifier.classify(question)
    pred_label = result["label"]
    confidence = round(result["probability"], 4)

    results.append({
        "question": question,
        "true": true_label,
        "pred": pred_label,
        "confidence": confidence
    })

    if pred_label == true_label:
        correct += 1

print("\n📊 Evaluation Results:")
for r in results:
    print("="*60)
    print(f"📝 Question: {r['question']}")
    print(f"✅ True Label: {r['true']} | 🔍 Predicted: {r['pred']} ({r['confidence']})")

accuracy = correct / len(test_data)
print("\n🎯 Accuracy:", round(accuracy * 100, 2), "%")

print("\n❌ Misclassified Samples:")
for r in results:
    if r["true"] != r["pred"]:
        print(f"- [{r['true']} ➜ {r['pred']}] {r['question']}")
