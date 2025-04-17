from keybert import KeyBERT
from typing import List

class KeywordExtractor:
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased"):
        self.model = KeyBERT(model=model_name)

    def extract(self, text: str, top_k: int = 5) -> List[str]:
        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=top_k
        )
        return [kw[0] for kw in keywords]  # only return keyword, no scores

if __name__ == "__main__":
    extractor = KeywordExtractor()
    text = "What is the statute of limitations for a breach of contract?"
    keywords = extractor.extract(text)
    print("Extraction Results:", keywords)
