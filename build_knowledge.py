from knowledge.loader import load_pdf_text, extract_text_from_images
from knowledge.index_builder import build_index
from knowledge.retriever import get_hybrid_retriever
from knowledge import init_environment

PDF_PATH = "./data/uscode-title23.pdf"
USE_OCR = True 

def main():
    init_environment()

    if USE_OCR:
        documents = [extract_text_from_images(PDF_PATH)]
    else:
        documents = load_pdf_text(PDF_PATH)

    index, nodes = build_index(documents)
    retriever = get_hybrid_retriever(index, nodes)

    question = "What is the definition of 'construction' under Title 23?"
    results = retriever.retrieve(question)

    print("\n📄 Retrieval Result：")
    for i, r in enumerate(results):
        print(f"\nResult {i+1} (Score: {r.score:.4f}):")
        print(r.node.get_content()[:300], "...\n")

if __name__ == "__main__":
    main()
