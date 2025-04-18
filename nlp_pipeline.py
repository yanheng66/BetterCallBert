from keybert import KeyBERT
import spacy
from typing import List, Tuple

class KeywordExtractor:
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased"):
        self.model = KeyBERT(model=model_name)

    def extract(self, text: str, top_k: int = 5) -> List[str]:
        """
        Extract keywords (including words and phrases)
        """
        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=top_k
        )
        return [kw[0] for kw in keywords]  # Only return keywords, discard scores

class EntityExtractor:
    def __init__(self, model_name="en_core_web_sm"):
        """
        Use spaCy to extract entities
        """
        self.nlp = spacy.load(model_name)

    def extract(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities from the text
        """
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

def expand_query(question: str, keyword_extractor: KeywordExtractor, entity_extractor: EntityExtractor) -> str:
    """
    Concatenate keywords and entities with the original question to improve retrieval recall
    """
    keywords = keyword_extractor.extract(question)
    entities = [txt for txt, _ in entity_extractor.extract(question)]

    # Remove duplicates and create additional query terms
    extra_terms = list(dict.fromkeys(keywords + entities))
    return " ".join([question] + extra_terms)
