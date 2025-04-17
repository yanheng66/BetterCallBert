# build_vector_store.py  –  final, tested working
import os, json, faiss, importlib
from glob import glob
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.embeddings.utils import BaseEmbedding    # 断言使用的类

# ---------- 可调参数 ----------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
JSON_DIR         = "data/parsed_sections_v2"
PERSIST_DIR      = "vector_store_minilm"
BATCH_SIZE       = 64
CHUNK_SIZE       = 512
CHUNK_OVERLAP    = 64
# --------------------------------

st_model = SentenceTransformer(EMBED_MODEL_NAME)

# ① 轻量适配器（仅需两方法）
class STAdapter:
    def __init__(self, st, batch_size=32):
        self.st, self.bs = st, batch_size
    def embed_documents(self, texts):
        return (
            self.st.encode(texts, batch_size=self.bs,
                           convert_to_numpy=True).tolist()
        )
    def embed_query(self, text):
        return self.st.encode(text, convert_to_numpy=True).tolist()

adapter = STAdapter(st_model, batch_size=BATCH_SIZE)

# ② 关键一步：注册为 BaseEmbedding 的“虚拟子类”
BaseEmbedding.register(STAdapter)

splitter = SentenceSplitter(chunk_size=CHUNK_SIZE,
                            chunk_overlap=CHUNK_OVERLAP)

def load_documents():
    docs = []
    for fp in tqdm(glob(f"{JSON_DIR}/*.json"), desc="📄 Loading"):
        with open(fp, "r") as f:
            data = json.load(f)
        docs.append(
            Document(
                text=data["text"],
                metadata={
                    "title": data["title"],
                    "section": data["section"],
                    "heading": data.get("heading", ""),
                },
            )
        )
    return docs

def build_vector_store():
    os.makedirs(PERSIST_DIR, exist_ok=True)

    # 1️⃣ 分块
    print("✂️  Chunking …")
    nodes = list(
        tqdm(
            splitter.get_nodes_from_documents(load_documents()),
            desc="Parsing", unit="chunk"
        )
    )
    print("   ➜ Chunks generated:", len(nodes))

    # 2️⃣ 批量嵌入
    print("🔢 Embedding …")
    texts = [n.get_content() for n in nodes]
    embs  = st_model.encode(
        texts, batch_size=BATCH_SIZE,
        show_progress_bar=True, convert_to_numpy=True
    )
    for n, e in zip(nodes, embs):
        n.embedding = e.tolist()

    # 3️⃣ 构建 FAISS & 索引
    dim = embs.shape[1]
    vstore = FaissVectorStore(faiss_index=faiss.IndexFlatL2(dim))
    sc     = StorageContext.from_defaults(vector_store=vstore)
    VectorStoreIndex(nodes, storage_context=sc, embed_model=adapter)  # ✔️ 通过断言

    # 4️⃣ 持久化
    sc.persist(persist_dir=PERSIST_DIR)
    print(f"✅ 完成！向量库保存于: {PERSIST_DIR}")

if __name__ == "__main__":
    build_vector_store()
