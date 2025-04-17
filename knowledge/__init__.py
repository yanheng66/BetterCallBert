import os
import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def init_environment():
    # ✅ disbale warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # ✅ load enviroment variable
    load_dotenv(override=True)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing Open Router API KEY")

    # ✅ LLM（OpenRouter API）
    Settings.llm = OpenAI(
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1", 
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=512
    )

    # ✅ lora embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="./legalbert-embedding-lora")

    # ✅ async problem
    nest_asyncio.apply()
    
    print("✅ Environment Initialization Complete")
