import os
import glob
import json
import faiss

from sentence_transformers import SentenceTransformer

from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# limit faiss to use single thread 
faiss.omp_set_num_threads(1)

def make_faiss_index(dim: int):
    print("✅ Use CPU FAISS Index（IndexFlatL2）")
    return faiss.IndexFlatL2(dim)

def load_documents(json_dir: str):
    """Read all under json_dir  .json，return Document list"""
    pattern = os.path.join(json_dir, '*.json')
    files = glob.glob(pattern)
    print(f"🔍 Under '{json_dir}' find {len(files)} jsons")
    docs = []
    for path in files:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        docs.append(Document(
            text=data.get('text', '').strip(),
            metadata={
                'title':   data.get('title', ''),
                'section': data.get('section', ''),
                'heading': data.get('heading', ''),
            }
        ))
    return docs

def build_vector_store(
    json_dir: str = 'data/parsed_sections_v2',
    persist_dir: str = 'vector_store_minilm',
    hf_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    chunk_size: int = 512,
    chunk_overlap: int = 64,
):
    # 1) Load Doc
    docs = load_documents(json_dir)
    if not docs:
        print("❌ Did not find any document，exit。")
        return

    # 2) Use sentence-transformers obtain embedding dimensions
    pt_model = SentenceTransformer(hf_model_name)
    dim = pt_model.get_sentence_embedding_dimension()
    print(f"🔢 Detected embedding dimensions：{dim}")

    # 3) Set up Global Settings
    Settings.embed_model = HuggingFaceEmbedding(model_name=hf_model_name)
    Settings.node_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 4) Construct FAISS Index & StorageContext
    cpu_index = make_faiss_index(dim)
    vector_store = FaissVectorStore(faiss_index=cpu_index)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    # 5) Auto Chunking
    print("🚀 stared Constructing Vector Index…")
    VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_ctx,
    )

    # 6) Persist to disk
    os.makedirs(persist_dir, exist_ok=True)
    storage_ctx.persist(persist_dir=persist_dir)
    print(f"✅ Vector store has saved to：'{persist_dir}'")
    print("📂 Folder includes：", os.listdir(persist_dir))

if __name__ == '__main__':
    build_vector_store()
