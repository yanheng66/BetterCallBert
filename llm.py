import os
from dotenv import load_dotenv
from llama_index.llms.openrouter import OpenRouter

# Load .env file
print("📂 Attempting to load .env file...")
load_dotenv()


def setup_llm():
    print("🔍 Checking for OPENROUTER_API_KEY environment variable...")
    api_key = os.getenv("OPENROUTER_API_KEY1")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found. Please check the .env file!")
        raise ValueError("❌ OPENROUTER_API_KEY not found in .env file.")

    print(f"✅ Successfully read API KEY: {api_key[:5]}...{api_key[-5:]}")
    print("🚀 Initializing OpenRouter LLM...")
    return OpenRouter(
        model="mistralai/mistral-7b-instruct",
        api_key=api_key,
        max_tokens=512,
        context_window=4096,
    )
