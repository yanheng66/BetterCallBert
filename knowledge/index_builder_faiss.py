# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.ingestion import IngestionPipeline
# from llama_index.core import VectorStoreIndex, Document, StorageContext
# import os
# import re

# def clean_ocr_text(text: str) -> str:
#     text = re.sub(r'[^\x00-\x7F]+', ' ', text)
#     text = re.sub(r'([?!.,])\1+', r'\1', text)
#     text = text.replace('\n', ' ')
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()

# # Vector base
# def build_index(documents, persist_dir="./vector_store/title23"):
#     # 1. Text Clean
#     documents = [Document(text=clean_ocr_text(doc.text), metadata=doc.metadata) for doc in documents]

#     # 2. Chunk
#     pipeline = IngestionPipeline(
#         transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=100)],
#     )
#     nodes = pipeline.run(documents=documents)

#     # 3. Clean Each Node
#     for node in nodes:
#         node.set_content(clean_ocr_text(node.get_content()))

#     # 4. Make Storage Directory
#     os.makedirs(persist_dir, exist_ok=True)
#     storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

#     # 5. Construct Vector base and persist it
#     index = VectorStoreIndex(nodes, storage_context=storage_context)
#     index.storage_context.persist()

#     print(f"✅ Retrieval Construction Complete，# of Nodes: {len(nodes)}")
#     print(f"✅ Vector base saved to：{persist_dir}")
#     return index, nodes

import os
import re
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_faiss_index(documents, persist_dir="./vector_store_faiss/title23"):
    # 文本清洗
    documents = [Document(text=clean_text(doc.text), metadata=doc.metadata) for doc in documents]

    # 切分为节点
    pipeline = IngestionPipeline(transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=100)
    ])
    nodes = pipeline.run(documents)

    # 构建 FAISS 向量库
    vector_store = FaissVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    # 保存向量库
    os.makedirs(persist_dir, exist_ok=True)
    vector_store.save(persist_dir)

    print(f"✅ FAISS 向量库已保存到：{persist_dir}")
    return index, nodes

