from nlp_pipeline import LegalNLPPipeline



questions = [
    "What are the consequences of homocide?",
    "Can I sue for defamation if someone insults me on Twitter?",
    "How long do I have to file a lawsuit in a medical malpractice case?"
]

nlp = LegalNLPPipeline()

for q in questions:
    result = nlp.analyze(q)
    print("="*50)
    print("📝 Question:", q)
    print("📂 Classifier:", result["category"], f"(Confidence {result['confidence']})")
    print("🔍 Keyword:", result["keywords"])
    print("🧾 Entity:", result["entities"])
