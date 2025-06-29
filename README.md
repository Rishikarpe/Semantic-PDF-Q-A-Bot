# 📄 Semantic PDF Q&A Bot

**Semantic PDF Q&A Bot** is a Python-based application that allows users to ask natural language questions about the contents of any PDF document and receive intelligent, context-aware answers. Leveraging semantic search and the power of local language models, it offers both a command-line interface and a user-friendly web app.

---

## 🔍 Key Features

- **Semantic Search with Embeddings**: Finds the most relevant text chunks using vector similarity.
- **Powered by Mistral 7B via Ollama**: Local LLM generates coherent, grounded answers.
- **PDF Upload Support**: Easily ask questions about your own documents.
- **RAG Pipeline Implementation**: Combines retrieval and generation for accurate results.
- **Dual Interface**: Use via CLI or Web UI.

---

## ⚙️ How It Works

This project follows a **Retrieval-Augmented Generation (RAG)** architecture:

1. **PDF Loading & Chunking**: Extracts text from PDFs using PyMuPDF and splits it into smaller chunks.
2. **Embedding Generation**: Uses `all-MiniLM-L6-v2` from Sentence-Transformers to convert chunks into vector embeddings.
3. **Vector Indexing**: FAISS indexes these embeddings for efficient semantic search.
4. **Semantic Search**: User questions are embedded and compared against the index to retrieve the most relevant chunks.
5. **Answer Generation**: Context chunks + user query are passed to Mistral-7B via Ollama for answer generation.

---

## 🧰 Tech Stack

| Component       | Technology                   |
|----------------|------------------------------|
| **Backend**     | Python, Flask                |
| **LLM**         | Mistral 7B via Ollama        |
| **Embeddings**  | Sentence-Transformers        |
| **Vector DB**   | FAISS                        |
| **PDF Parsing** | PyMuPDF (`fitz`)             |
| **Frontend**    | HTML, Tailwind CSS           |

---

## 🚀 Getting Started

### 📦 Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running
- Pull the Mistral 7B model:
  ```bash
  ollama pull mistral:7b

---

## Installation
git clone https://github.com/rishikarpe/Semantic-PDF-Q-A-Bot.git
cd Semantic-PDF-Q-A-Bot
pip install flask PyMuPDF sentence-transformers faiss-cpu numpy ollama werkzeug

---

## Running the Application
- 1. Command-Line Interface (CLI)
Works with a predefined sample.pdf:
python main.py

- 2. Web Application
Provides a web UI to upload any PDF:
python app2.py

Then open your browser and visit:
http://127.0.0.1:5000

---

# Use Cases
Academic research document analysis
Legal or policy document summarization
Internal report querying
Contract review or compliance checks
