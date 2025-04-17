import os
import json
import pickle
from glob import glob
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_documents(json_dir):
    documents = []
    paths = glob(os.path.join(json_dir, "*.json"))
    for path in tqdm(paths, desc="📄 Loading JSON files"):
        with open(path, "r") as f:
            data = json.load(f)
        documents.append(Document(
            text=data["text"],
            metadata={
                "title": data["title"],
                "section": data["section"],
                "heading": data["heading"]
            }))
    return documents

def build_vector_store(documents, persist_dir="./vector_store_minilm"):
    os.makedirs(persist_dir, exist_ok=True)

    print("✂️ Chunking documents ...")
    pipeline = IngestionPipeline(transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=64)])
    nodes = pipeline.run(documents=documents)

    print("🔢 Embedding nodes ...")
    embedded_nodes = []
    for i, node in enumerate(tqdm(nodes, desc="🔐 Encoding")):
        node.embedding = embed_model.encode(node.get_content()).tolist()
        embedded_nodes.append(node)


        if i > 0 and i % 1000 == 0:
            with open("embedded_nodes.pkl", "wb") as f:
                pickle.dump(embedded_nodes, f)
            print(f"💾 Auto-saved {i} embeddings to embedded_nodes.pkl")

    # Final Save
    with open("embedded_nodes.pkl", "wb") as f:
        pickle.dump(embedded_nodes, f)
    print("✅ Saved all embeddings to embedded_nodes.pkl")

    print("💾 Saving to FAISS vector store ...")
    embedding_dim = len(embedded_nodes[0].embedding)
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(nodes=embedded_nodes, storage_context=storage_context)
    index.storage_context.persist(persist_dir=persist_dir)

    print(f"✅ All done! Total nodes: {len(embedded_nodes)}")
    print(f"✅ FAISS vector base saved to: {persist_dir}")

# === Main ===
if __name__ == "__main__":
    docs = load_documents("data/parsed_sections_v2")
    build_vector_store(docs)

# import os
# import json
# from glob import glob
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer

# from llama_index.core import Document, StorageContext, VectorStoreIndex
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.ingestion import IngestionPipeline
# from llama_index.vector_stores.faiss import FaissVectorStore
# import faiss


# EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
# JSON_DIR      = "data/parsed_sections_v2"
# PERSIST_DIR   = "vector_store_minilm"
# BATCH_SIZE    = 64
# CHUNK_SIZE    = 512
# CHUNK_OVERLAP = 64
# # =================

# embed_model = SentenceTransformer(EMBED_MODEL)
# splitter    = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# def load_documents(json_dir):
#     docs = []
#     for path in tqdm(glob(os.path.join(json_dir, "*.json")),
#                      desc="📄 Loading JSON files"):
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         docs.append(Document(
#             text=data["text"],
#             metadata={
#                 "title":   data["title"],
#                 "section": data["section"],
#                 "heading": data["heading"],
#             }
#         ))
#     return docs

# def build_vector_store(documents, persist_dir=PERSIST_DIR):
#     os.makedirs(persist_dir, exist_ok=True)

#     # 1️⃣ Chunk everything into a list
#     print("✂️ Chunking documents …")
#     pipeline = IngestionPipeline(transformations=[splitter])
#     nodes = list(pipeline.run(documents=documents))
#     print(f"   ➜ Total chunks: {len(nodes)}")

#     # 2️⃣ Bulk‑encode all chunk texts
#     print("🔢 Bulk embedding chunks …")
#     texts = [n.get_content() for n in nodes]
#     embeddings = embed_model.encode(
#         texts,
#         batch_size=BATCH_SIZE,
#         show_progress_bar=True,
#         convert_to_numpy=True
#     )
#     for node, emb in zip(nodes, embeddings):
#         node.embedding = emb.tolist()

#     # 3️⃣ Build FAISS and persist
#     print("💾 Building and persisting FAISS vector store …")
#     dim = embeddings.shape[1]
#     faiss_index  = faiss.IndexFlatL2(dim)
#     vector_store = FaissVectorStore(faiss_index=faiss_index)
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)

#     index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
#     index.storage_context.persist(persist_dir=persist_dir)

#     print(f"✅ Done! {len(nodes)} chunks embedded and saved to '{persist_dir}'")

# if __name__ == "__main__":
#     docs = load_documents(JSON_DIR)
#     build_vector_store(docs)
