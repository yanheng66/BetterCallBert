import os
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.faiss import FaissVectorStore

# === Configure path ===
VECTOR_STORE_DIR = "vector_store_minilm"
Settings.llm = None

# === Load embedding model ===
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Load FAISS vector index ===
print("📦 Loading vector store...")
storage_context = StorageContext.from_defaults(
    persist_dir=VECTOR_STORE_DIR,
    vector_store=FaissVectorStore.from_persist_dir(VECTOR_STORE_DIR),
)

index = load_index_from_storage(storage_context, embed_model=embed_model)

# === Construct QueryEngine ===
retriever = index.as_retriever(similarity_top_k=3)
query_engine = RetrieverQueryEngine(retriever=retriever)

# === Get user's input query ===
QUERY = input("Input your questions: ")

# === Execute the query and output the result ===
print(f"\n🔍 Query: {QUERY}")
response = query_engine.query(QUERY)

# === Output the query result ===
print("\n📚 Relevant legal texts:")
print(response)
