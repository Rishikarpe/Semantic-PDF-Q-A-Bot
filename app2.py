from flask import Flask, render_template, request
import os
import fitz  # PyMuPDF
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import ollama

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Check if uploaded file is PDF
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extract and chunk PDF text
def load_pdf_chunks(path, chunk_size=500):
    doc = fitz.open(path)
    text = "".join([page.get_text() for page in doc])
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Ask mistral via ollama
def ask_mistral(question, context):
    prompt = f"""You are a helpful assistant. Use the context below to answer.

Context:
{context}

Question: {question}
Answer:"""
    response = ollama.chat(model="mistral:7b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    question = ""

    if request.method == "POST":
        question = request.form["question"]
        file = request.files["pdf_file"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Load and embed chunks
            chunks = load_pdf_chunks(filepath)
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(chunks)
            dim = embeddings.shape[1]

            # Build FAISS index
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)

            # Search for top-k relevant chunks
            query_embedding = model.encode([question])
            D, I = index.search(np.array(query_embedding), k=3)
            context = "\n\n".join(chunks[i] for i in I[0])

            # Get answer from Mistral
            answer = ask_mistral(question, context)

    return render_template("main.html", answer=answer, question=question)

if __name__ == "__main__":
    app.run(debug=True)
