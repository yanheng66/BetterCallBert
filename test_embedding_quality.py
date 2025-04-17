from sentence_transformers import SentenceTransformer
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, load_index_from_storage

vector_store = FaissVectorStore.from_persist_dir("vector_store_minilm")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = load_index_from_storage(storage_context)

retriever = index.as_retriever(similarity_top_k=3)

query = "What is the definition of marriage under U.S. Code?"
results = retriever.retrieve(query)

print("\n🔍 Top 3 Results:")
for i, r in enumerate(results, 1):
    snippet = r.node.get_content()[:300].replace("\n", " ")
    print(f"Result {i} (Score: {r.score:.4f}): {snippet}...")
