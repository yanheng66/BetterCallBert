from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.faiss import FaissVectorStore

# ✅ If your version supports BM25Retriever
from llama_index.retrievers.bm25 import BM25Retriever


# ✅ Original Vector Retriever (unchanged)
class Retriever:
    def __init__(self, vector_store_dir: str = "vector_store_minilm"):
        self.vector_store_dir = vector_store_dir
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.embed_model = Settings.embed_model

        print("📦 Loading vector store (vector retriever)...")
        storage_context = StorageContext.from_defaults(
            persist_dir=self.vector_store_dir,
            vector_store=FaissVectorStore.from_persist_dir(self.vector_store_dir),
        )

        self.index = load_index_from_storage(storage_context, embed_model=self.embed_model)
        self.retriever = self.index.as_retriever(similarity_top_k=3)
        self.query_engine = RetrieverQueryEngine(retriever=self.retriever)

    def retrieve(self, query: str):
        print(f"\n🔍 [VectorRetriever] Query: {query}")
        return self.query_engine.query(query)


# ✅ New BM25 Retriever class (can be used for keyword-based retrieval)
class BM25RetrieverWrapper:
    def __init__(self, vector_store_dir: str = "vector_store_minilm"):
        print("📦 Initializing BM25 Retriever...")

        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.embed_model = Settings.embed_model

        storage_context = StorageContext.from_defaults(
            persist_dir=vector_store_dir,
            vector_store=FaissVectorStore.from_persist_dir(vector_store_dir),
        )

        self.index = load_index_from_storage(storage_context, embed_model=self.embed_model)
        self.retriever = BM25Retriever.from_defaults(self.index)
        self.query_engine = RetrieverQueryEngine(retriever=self.retriever)

    def retrieve(self, query: str):
        print(f"\n🔍 [BM25Retriever] Query: {query}")
        return self.query_engine.query(query)
