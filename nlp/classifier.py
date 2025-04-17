from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from pathlib import Path

class LegalClassifier:
    def __init__(self, model_name="legalbert-lora-finetuned/final", label_map=None):
        model_path = Path(model_name).resolve()
        if not model_path.is_dir():
            raise ValueError(f"🚫 The model path is not a valid directory: {model_path}")

        print(f"✅ Loading classifier model from: {model_path}")
        self.label_map = label_map or {0: "Contract", 1: "Tort", 2: "Criminal", 3: "Other"}

        # ✅ 关键：确保传入的是 Path 对象，而不是字符串
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
            local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_path,
            local_files_only=True
        )
        self.model.eval()

    def classify(self, question: str):
        inputs = self.tokenizer(question, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze()
            pred = torch.argmax(probs).item()
        
        return {
            "label": self.label_map[pred],
            "probability": probs[pred].item(),
            "all_probs": {self.label_map[i]: round(p.item(), 4) for i, p in enumerate(probs)}
        }



if __name__ == "__main__":
    classifier = LegalClassifier(label_map={0: "Contract", 1: "Tort", 2: "Criminal", 3: "Other"})
    q = "What are the consequences of homocide?"
    result = classifier.classify(q)
    print("Prediction Category:", result["label"])
    print("Probability Distribution:", result["all_probs"])
    print("✅ Model test complete")
