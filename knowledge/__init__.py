import os
import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def init_environment():
    load_dotenv(override=True)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing Open Router API KEY")

    Settings.llm = OpenAI(
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1", 
        model="mistralai/mistral-7b-instruct",
        temperature=0.2,
        max_tokens=512
    )
    Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

    nest_asyncio.apply()
    print("Enviroment Initialization Complete")

