from nlp_pipeline import KeywordExtractor, EntityExtractor, expand_query
from retriever import Retriever, BM25RetrieverWrapper
from llm import setup_llm
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response


def legal_qa(question: str) -> dict:
    """
    Use both vector retrieval and BM25 retrieval to generate answers for the same legal question.
    The returned structure includes two answers with their respective citations.
    """

    # Initialize components
    keyword_extractor = KeywordExtractor()
    entity_extractor = EntityExtractor()
    llm = setup_llm()

    # Expand the query
    expanded_query = expand_query(question, keyword_extractor, entity_extractor)

    # Initialize two retrievers
    retrievers = {
        "vector": Retriever(),
        "bm25": BM25RetrieverWrapper()
    }

    # Prompt template
    template_str = """
You are a professional legal assistant. Using only the legal context below, answer the question in 1-3 concise sentences.

Do not repeat the question. Do not include citations, file metadata, or irrelevant information.

---

Context:
{context}

---

Question:
{question}

---

Answer:
"""
    prompt_template = PromptTemplate(template_str)

    # Final results dictionary
    results = {}

    for name, retriever in retrievers.items():
        response = retriever.retrieve(expanded_query)

        sources = []
        if isinstance(response, Response):
            sources = [str(node.node.text).strip() for node in response.source_nodes[:3]]

        formatted_prompt = prompt_template.format(
            context="\n".join(sources),
            question=question
        )

        llm_response = llm.complete(formatted_prompt)

        results[name] = {
            "answer": llm_response.text.strip(),
            "sources": sources
        }

    return results


# ✅ Test entry point
if __name__ == "__main__":
    user_question = input("Enter your legal question: ")
    result = legal_qa(user_question)

    print("\n📙 Vector-based retrieval answer:")
    print(result["vector"]["answer"])
    for i, src in enumerate(result["vector"]["sources"], 1):
        print(f"[{i}] {src}")

    print("\n📘 BM25 retrieval answer:")
    print(result["bm25"]["answer"])
    for i, src in enumerate(result["bm25"]["sources"], 1):
        print(f"[{i}] {src}")
