from nlp.classifier import LegalClassifier
from nlp.keyword_extractor import KeywordExtractor
from nlp.entity_extractor import EntityExtractor


class LegalNLPPipeline:
    def __init__(self):
        self.classifier = LegalClassifier(label_map={0: "Contract", 1: "Tort", 2: "Criminal", 3: "Other"})
        self.keyword_extractor = KeywordExtractor()
        self.entity_extractor = EntityExtractor()

    def analyze(self, question: str) -> dict:
        # Classification
        classification = self.classifier.classify(question)
        
        # Keyword Extraction
        keywords = self.keyword_extractor.extract(question)

        # Entity Recog
        entities = self.entity_extractor.extract(question)

        return {
            "question": question,
            "category": classification["label"],
            "confidence": round(classification["probability"], 4),
            "classification_probs": classification["all_probs"],
            "keywords": keywords,
            "entities": entities
        }

if __name__ == "__main__":
    pipeline = LegalNLPPipeline()
    question = "Can I sue Amazon for a breach of contract in California?"
    result = pipeline.analyze(question)

    print("\n🧠 NLP Analyze Result")
    print("📌 Original Question:", result["question"])
    print("📂 Classification:", result["category"], f"(置信度 {result['confidence']})")
    print("🔍 Keyword:", result["keywords"])
    print("🧾 Entity:", result["entities"])
