# build_knowledge.py
# -----------------------------------------------------------
# 功能：读取法规 PDF → 构建混合检索索引 → NLP 扩充查询 → 检索 →
#      将上下文交给 LLM 生成法律专业回答
# -----------------------------------------------------------

from pathlib import Path
from knowledge.loader import load_pdf_text, extract_text_from_images
from knowledge.index_builder import build_index
from knowledge.retriever import get_hybrid_retriever
from knowledge import init_environment

from nlp.nlp_pipeline import LegalNLPPipeline          # ★ 新增：引入 NLP 管道
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import ChatResponse
from llama_index.core import Settings

# ======== 全局配置 ========
PDF_PATH = "./data/uscode-title23.pdf"   # 示例：美国联邦法典 Title 23
USE_OCR  = True                          # True → PDF 转图片后 OCR；False → 直接读取文本
TOP_K    = 5                             # 检索返回的段落数量

# ======== 初始化环境 ========
init_environment()                       # OpenRouter LLM + LoRA 嵌入模型
nlp_pipeline = LegalNLPPipeline()        # NLP 管道（分类 / 关键词 / 实体）

# -----------------------------------------------------------
# 函数：用 NLP 结果扩充查询（Query Expansion）
# -----------------------------------------------------------
def expand_query(question: str) -> str:
    """
    将 NLP 提取的关键词和实体与原问题拼接，提升检索召回率
    """
    analysis  = nlp_pipeline.analyze(question)
    keywords  = analysis["keywords"]                     # KeyBERT 关键词
    entities  = [txt for txt, _ in analysis["entities"]] # spaCy 实体文本

    # 去重后组成额外查询词
    extra_terms = list(dict.fromkeys(keywords + entities))
    # 返回“原始问题 + 关键词 + 实体”的合并字符串
    return " ".join([question] + extra_terms)

# -----------------------------------------------------------
# 函数：将检索上下文与问题拼接，交给 LLM 获取最终回答
# -----------------------------------------------------------
def answer_question(llm, question: str, contexts):
    context_text = "\n\n".join([r.node.get_content() for r in contexts])

    prompt = PromptTemplate(
        "You are a legal assistant. Based on the following legal context, answer the question:\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer in clear legal English:"
    )
    full_prompt = prompt.format(context=context_text, question=question)
    response: ChatResponse = llm.complete(full_prompt)
    return response.text.strip()

# -----------------------------------------------------------
# 主流程
# -----------------------------------------------------------
def main():
    # 1) 读取/解析 PDF
    if USE_OCR:
        documents = [extract_text_from_images(PDF_PATH)]
    else:
        documents = load_pdf_text(PDF_PATH)

    # 2) 构建索引 & 检索器
    index, nodes = build_index(documents)
    retriever    = get_hybrid_retriever(index, nodes)

    # 3) 用户提问
    question = "What is the definition of 'construction' under Title 23?"  # 示例问题

    # 4) 扩充查询并检索
    expanded_query = expand_query(question)
    results        = retriever.retrieve(expanded_query)[:TOP_K]

    print(f"\n📄 检索结果（Top {TOP_K}）：")
    for i, r in enumerate(results):
        snippet = r.node.get_content()[:300].replace("\n", " ")
        print(f"\nResult {i+1} (Score: {r.score:.4f}): {snippet} ...")

    # 5) 交给 LLM 生成最终回答
    answer = answer_question(Settings.llm, question, results)
    print("\n🧠 AI 回答：\n", answer)

# -----------------------------------------------------------
# 脚本入口
# -----------------------------------------------------------
if __name__ == "__main__":
    main()
