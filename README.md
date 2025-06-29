# Semantic PDF Q&A Bot

This repository contains a Python application that allows you to ask questions about the content of a PDF document and receive intelligent answers. The bot uses a semantic search approach with vector embeddings to find the most relevant context within the PDF and then uses the Mistral-7B language model via Ollama to generate a coherent answer.

The project includes two main implementations:
1.  A command-line interface (CLI) for interacting with a predefined PDF.
2.  A Flask web application that allows users to upload their own PDF and ask questions through a web UI.

## How It Works

The process follows a Retrieval-Augmented Generation (RAG) pipeline:

1.  **PDF Loading and Chunking**: The text content is extracted from the PDF using `PyMuPDF`. This text is then divided into smaller, manageable chunks.
2.  **Embedding Generation**: Each text chunk is converted into a numerical vector representation (embedding) using the `all-MiniLM-L6-v2` model from `sentence-transformers`.
3.  **Vector Indexing**: The generated embeddings are stored in a `FAISS` (Facebook AI Similarity Search) index. This creates a searchable vector database that allows for efficient similarity searches.
4.  **Semantic Search**: When a user asks a question, the question is also converted into an embedding. FAISS is then used to search the index for the text chunks with embeddings most similar to the question's embedding.
5.  **Answer Generation**: The most relevant text chunks (the "context") are passed along with the original question to the `mistral:7b` model via Ollama. The model uses this context to generate a helpful and accurate answer.

## Technology Stack

-   **Backend**: Python, Flask
-   **LLM**: Ollama (Mistral 7B)
-   **Embeddings**: Sentence-Transformers
-   **Vector Search**: FAISS
-   **PDF Parsing**: PyMuPDF (`fitz`)
-   **Frontend**: HTML, Tailwind CSS

## Getting Started

### Prerequisites

-   Python 3.8+
-   [Ollama](https://ollama.com/) installed and running.
-   The Mistral model pulled via Ollama:
    ```sh
    ollama pull mistral:7b
    ```

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/rishikarpe/Semantic-PDF-Q-A-Bot.git
    cd Semantic-PDF-Q-A-Bot
    ```

2.  Install the required Python packages. It is recommended to use a virtual environment.
    ```sh
    pip install flask PyMuPDF sentence-transformers faiss-cpu numpy ollama werkzeug
    ```

### Running the Application

You can run either the command-line version or the web application.

#### 1. Command-Line Interface (CLI)

This version works with a hardcoded `sample.pdf` file in the root directory.

```sh
python main.py
```

You will be prompted to ask questions in the terminal. Type `exit` to quit.

#### 2. Web Application

This version provides a user interface to upload your own PDF file.

```sh
python app2.py
```

Navigate to `http://127.0.0.1:5000` in your web browser. You can then upload a PDF, type your question, and receive an answer directly on the page.

## Deployment

This project includes a `render.yaml` file for easy deployment as a web service on [Render](https://render.com/). The configuration specifies the build and start commands, ensuring a smooth deployment process.
