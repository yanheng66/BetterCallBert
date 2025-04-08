import os
import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer


def init_environment():
    load_dotenv(override=True)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing Open Router API KEY")

    Settings.llm = OpenAI(
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1", 
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=512
    )

    # ✅ Load Lora Fine Tuned Embedding Model
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("./legalbert-embedding-lora")
    Settings.embed_model = HuggingFaceEmbedding(model_name="./legalbert-embedding-lora")



    nest_asyncio.apply()
    print("✅ Environment Initialization Complete")

