from nlp.nlp_pipeline import LegalNLPPipeline



questions = [
    "What are the consequences of breaching a non-compete clause?",
    "Can I sue for defamation if someone insults me on Twitter?",
    "How long do I have to file a lawsuit in a medical malpractice case?"
]

nlp = LegalNLPPipeline()

for q in questions:
    result = nlp.analyze(q)
    print("="*50)
    print("📝 问题:", q)
    print("📂 分类:", result["category"], f"(置信度 {result['confidence']})")
    print("🔍 关键词:", result["keywords"])
    print("🧾 实体:", result["entities"])
