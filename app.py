from flask import Flask, render_template, request
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

app = Flask(__name__)

# --- Load and Chunk PDF ---
def load_pdf(path, chunk_size=500):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# --- Setup FAISS Index and Embeddings ---
chunks = load_pdf("sample.pdf")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(chunks)
dimension = embeddings.shape[1]

faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(embeddings))

# --- Ask Mistral via Ollama ---
def ask_mistral(question, context):
    prompt = f"""You are a helpful assistant. Answer the question using the context.

Context:
{context}

Question: {question}
Answer:"""
    response = ollama.chat(model="mistral:7b", messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()

# --- Flask Route ---
@app.route("/", methods=["GET", "POST"])
def qa_page():
    answer = ""
    question = ""
    if request.method == "POST":
        question = request.form["question"]
        q_embed = embed_model.encode([question])
        D, I = faiss_index.search(np.array(q_embed), k=3)
        context = "\n\n".join(chunks[i] for i in I[0])
        answer = ask_mistral(question, context)
    return render_template("index.html", answer=answer, question=question)

# --- Run the Flask App ---
if __name__ == "__main__":
    app.run(debug=True)
