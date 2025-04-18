# BetterCallBert: Legal AI Assistant

A specialized Retrieval Augmented Generation (RAG) system for answering legal questions using the U.S. Code as a knowledge base.

## Overview

BetterCallBert combines advanced NLP techniques, legal-domain BERT models, and RAG to provide accurate, contextually relevant answers to legal questions. The system:

1. Retrieves relevant legal texts from the U.S. Code using both semantic search and keyword-based search
2. Processes legal texts with domain-adapted transformer models
3. Generates concise, targeted answers based on the retrieved legal context

## Key Features

- **Dual Retrieval System:** Compare vector-based and BM25 retrieval methods side-by-side
- **Domain-Specific Models:** Uses LegalBERT for keyword extraction and NLP tasks
- **Query Expansion:** Automatically enhances queries with extracted keywords and entities
- **U.S. Code Knowledge Base:** Structured legal text collection from GovInfo API
- **Interactive Web Interface:** User-friendly Streamlit UI for asking legal questions

## System Architecture

BetterCallBert consists of several integrated components:

### Data Processing Pipeline
- Fetches legal texts from the U.S. Government API (GovInfo) 
- Parses documents into structured JSON format
- Builds FAISS vector index for efficient semantic search

### NLP Components
- Keyword extraction using KeyBERT (built on LegalBERT)
- Entity recognition with SpaCy
- Query expansion to improve retrieval accuracy
- LoRA-based model fine-tuning capabilities for legal text classification and embeddings

### Retrieval System
- Dense vector retrieval with FAISS and sentence transformers
- Sparse BM25 retrieval for traditional keyword search
- Side-by-side comparison of both retrieval methods

### Answer Generation
- LLM integration with Mistral-7B-Instruct model via OpenRouter
- Context-aware answer generation from retrieved legal texts
- Focused responses with legal citations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BetterCallBert.git
cd BetterCallBert

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Set up environment variables
cp .env.example .env
# Edit .env file to add your GovInfo API key as API_KEY and 
# your OpenRouter API key as OPENROUTER_API_KEY1
```

## Usage

### 1. Data Preparation
```bash
# Fetch and parse U.S. Code documents
python fetch_and_parse_legal_texts.py

# Build vector store from legal documents
python build_vector_store.py
```

### 2. Model Fine-tuning (Optional)
```bash
# Fine-tune embedding model with LoRA (requires legal_embedding_pairs.jsonl dataset)
python knowledge/train_lora_embedding.py

# Train LoRA-enhanced legal text classifier (requires legal_title_qa_dataset_5300.jsonl)
python nlp/train_lora_classifier.py

# Test embedding quality
python test_embedding_quality.py
```

### 3. Run the Web App
```bash
# Start Streamlit app
python -m streamlit run app.py
```

Then open your browser to http://localhost:8501

## Example Queries

- "What are the requirements for copyright protection?"
- "Explain the process for filing a patent application"
- "What constitutes fair use under copyright law?"
- "What are the penalties for tax evasion?"
- "What is the legal definition of bankruptcy?"

## Technical Details

### Dependencies

- **Core Frameworks:**
  - llama-index: RAG framework
  - transformers & sentence-transformers: For text embeddings
  - FAISS: Vector search engine
  - PEFT: Parameter-Efficient Fine-Tuning
  
- **NLP Components:**
  - KeyBERT: Keyword extraction
  - SpaCy: Named entity recognition
  
- **UI and Integration:**
  - Streamlit: Web interface
  - OpenRouter: LLM API access (Mistral-7B-Instruct)
 
### Sreamlit Screenshot
<img width="789" alt="Streamlit" src="https://github.com/user-attachments/assets/22b7c5f4-51e5-41ca-825a-c5b240e2a994" />

### Data Flow

1. User submits a question through the UI
2. Query is expanded with relevant keywords and entities
3. Expanded query is used for both vector-based retrieval (FAISS) and keyword-based retrieval (BM25)
4. Retrieved contexts from both methods are processed separately by the LLM
5. Both retrieval methods' results are displayed side-by-side in the UI

### Configuration Notes

- The system currently forces CPU usage for embeddings to avoid CUDA tensor errors
- API keys for GovInfo and OpenRouter must be configured in a .env file
- The fine-tuned models are not automatically integrated into the main pipeline

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- This project uses the U.S. Code from the GovInfo API
- Built with LegalBERT and other models from Hugging Face
- Uses the Mistral-7B-Instruct model via OpenRouter
