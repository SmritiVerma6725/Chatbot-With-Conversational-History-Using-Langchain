Overview
This repository contains a suite of advanced conversational AI chatbot projects built using the latest LLM (Large Language Model) and Retrieval-Augmented Generation (RAG) techniques. The projects leverage LangChain, Chroma, HuggingFace Embeddings, Gradio, and other modern tools to enable context-aware, memory-augmented, and document-grounded conversations. Each project demonstrates a unique aspect of conversational AI, from simple chatbots to multi-turn Q&A with external knowledge retrieval and persistent chat history.

Features
Conversational Q&A with Memory:
Chatbots that remember previous interactions and incorporate chat history for context-aware responses.

Retrieval-Augmented Generation (RAG):
Combines LLMs with vector-based document retrieval for accurate, grounded answers over custom knowledge bases.

Document Loading & Indexing:
Supports ingestion and chunking of PDFs, web pages, and other sources using DirectoryLoader, PyPDFLoader, and WebBaseLoader.

Vector Stores:
Integrates Chroma and FAISS for fast, persistent vector search.

Prompt Engineering:
Customizable prompt templates for role, style, and safety of chatbot responses.

Multi-Session Support:
Isolates chat history per user/session for concurrent conversations.

Flexible UI:
Includes both terminal-based and Gradio web interfaces.

Projects Included
1. Conversational Q&A Chatbot
Multi-turn chat with memory using LangChain's RunnableWithMessageHistory.

Supports both chain-based and agent-based retrieval.

Integrates Chroma/FAISS vector stores with HuggingFace sentence embeddings.

Example: Answering questions about a blog post or custom PDFs.

2. Ollama Chatbot
Local LLM inference using Ollama for privacy and low latency.

Streamlit interface for interactive chat.

Adjustable model parameters (temperature, max tokens, etc.).

3. Q&A Chatbot with OpenAI
Uses OpenAI's GPT models via LangChain for high-quality answers.

Secure API key management and prompt engineering.

Streamlit UI with model selection and parameter tuning.

Quick Start
Prerequisites
Python 3.10+

pip

(Optional) CUDA-enabled GPU for local inference

Installation
bash
git clone https://github.com/yourusername/conversational-rag-chatbot.git
cd conversational-rag-chatbot
pip install -r requirements.txt
Environment Variables
Create a .env file in the project root and add your API keys:

text
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
Usage
Run Conversational Q&A Chatbot
bash
python Chatbot.py
Run Ollama Chatbot
bash
streamlit run OllamaChatbot.py
Run Q&A Chatbot with OpenAI
bash
streamlit run QAChatbot.py
Project Structure
text
.
├── Chatbot.ipynb           # LLM-powered chatbot with memory
├── ConversationalQ-A.ipynb # Q&A chatbot with retrieval and chat history
├── OllamaChatbot.py        # Local LLM chatbot (Ollama + Streamlit)
├── QAChatbot.py            # OpenAI-powered Q&A chatbot (Streamlit)
├── requirements.txt        # All dependencies
├── Data/                   # Directory for PDF and data files
├── .env                    # Environment variables (not committed)
└── README.md               # This file
Key Concepts
Memory Management:
Uses session IDs and message history to provide context over multiple turns.

Retrieval Chains & Agents:
Chain-based retrieval always executes a search step; agents can dynamically decide when/how to retrieve.

Document Chunking:
Uses RecursiveCharacterTextSplitter for optimal chunk size and overlap, balancing context and granularity.

Vector Search:
Chroma and FAISS enable fast, approximate nearest neighbor search for relevant document chunks.

Prompt Templates:
System and user prompts are engineered for clarity, safety, and role alignment.

Example: Creating a Vector Database from PDFs
python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

loader = DirectoryLoader("Data/", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = splitter.split_documents(documents)
embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
vector_db.persist()
Best Practices
Environment Security:
Store API keys in .env files, never in code.

Error Handling:
All user-facing functions include try/except blocks for graceful failure and debugging.

Scalability:
Vector stores are persistent and can be scaled horizontally for large document corpora.

Extensibility:
Easily add new loaders, retrievers, or UI components as needed.





