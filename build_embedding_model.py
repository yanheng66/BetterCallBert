from sentence_transformers import SentenceTransformer, models

# use HuggingFace pretrained model（original LegalBERT）
hf_model_name = "nlpaueb/legal-bert-base-uncased"

# Construct SentenceTransformer structure
word_embedding_model = models.Transformer(hf_model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Save to embedding format
embedding_model.save("./legalbert-embedding")

print("✅ LegalBERT Embedding Model Complete，Save Path：./legalbert-embedding")
