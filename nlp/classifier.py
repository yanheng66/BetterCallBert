from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class LegalClassifier:
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased", label_map=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_map))
        self.label_map = label_map or {0: "Contract", 1: "Tort", 2: "Criminal", 3: "Other"}
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
    q = "Is a verbal agreement enforceable under employment law?"
    result = classifier.classify(q)
    print("Prediction Category:", result["label"])
    print("Probability Distribution:", result["all_probs"])
