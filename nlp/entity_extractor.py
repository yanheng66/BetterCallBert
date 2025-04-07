import spacy
from typing import List, Tuple

class EntityExtractor:
    def __init__(self, model_name="en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def extract(self, text: str) -> List[Tuple[str, str]]:
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

if __name__ == "__main__":
    extractor = EntityExtractor()
    text = "Can I sue Amazon for negligence under California law?"
    entities = extractor.extract(text)
    print("Entity Results:", entities)
