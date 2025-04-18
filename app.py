import os
import streamlit as st
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llm import setup_llm

# 💡 Disable huggingface fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 💡 Force use of CPU to solve meta tensor error
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ✅ Set default embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# ✅ Set default LLM
Settings.llm = setup_llm()

# ✅ Print debug information to confirm successful setup
print("✅ Current default embedding model:", Settings.embed_model)
print("✅ Current default language model:", Settings.llm)

# ==== Main Project Logic ====
from legalbert_qa import legal_qa

# Page settings
st.set_page_config(page_title="Legal AI Assistant", page_icon="⚖️")

# Page title and description
st.title("⚖️ Legal AI Assistant")
st.markdown("Ask your legal question and get an AI-generated answer grounded in legal documents.")

# User input question
user_input = st.text_area("📌 Enter your legal question:")

# Trigger query after clicking submit
if st.button("Submit") and user_input.strip():
    with st.spinner("Generating answer..."):
        try:
            result = legal_qa(user_input)
            st.success("✅ Answers generated successfully!")

            # Use Streamlit two-column layout
            col1, col2 = st.columns(2)

            # 🟦 Vector-based retrieval answer
            with col1:
                st.subheader("📙 Vector-based Answer")
                st.markdown(result["vector"]["answer"])
                if result["vector"].get("sources"):
                    st.markdown("#### 🔍 Sources")
                    for i, src in enumerate(result["vector"]["sources"], 1):
                        st.markdown(f"**[{i}]** {src}")

            # 🟨 BM25 retrieval answer
            with col2:
                st.subheader("📘 BM25 Answer")
                st.markdown(result["bm25"]["answer"])
                if result["bm25"].get("sources"):
                    st.markdown("#### 🔍 Sources")
                    for i, src in enumerate(result["bm25"]["sources"], 1):
                        st.markdown(f"**[{i}]** {src}")

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
