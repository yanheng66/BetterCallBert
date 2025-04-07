from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
import re
from llama_index.core import Document



def clean_ocr_text(text: str) -> str:
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'([?!.,])\1+', r'\1', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_index(documents):
    documents = [Document(text=clean_ocr_text(doc.text), metadata=doc.metadata) for doc in documents]

    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=100)],
    )
    nodes = pipeline.run(documents=documents)

    for node in nodes:
        node.set_content(clean_ocr_text(node.get_content()))

    index = VectorStoreIndex(nodes)
    print("✅ Retrieval Construction Complete，# of Nodes:", len(nodes))
    return index, nodes
