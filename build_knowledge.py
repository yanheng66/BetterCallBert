from knowledge.loader import load_pdf_text, extract_text_from_images
from knowledge.index_builder import build_index
from knowledge.retriever import get_hybrid_retriever
from knowledge import init_environment
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import ChatResponse
from llama_index.core import Settings

def answer_question(llm, query, contexts):
    context_texts = "\n\n".join([r.node.get_content() for r in contexts])
    prompt = PromptTemplate(
        "You are a legal assistant. Based on the following legal context, answer the question:\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer in clear legal English:"
    )
    full_prompt = prompt.format(context=context_texts, question=query)
    response: ChatResponse = llm.complete(full_prompt)
    return response.text

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

    answer = answer_question(Settings.llm, question, results)
    print("🧠 AI Summary Answer:")
    print(answer)

if __name__ == "__main__":
    main()
