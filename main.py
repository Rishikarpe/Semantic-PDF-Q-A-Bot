import fitz  
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

# --- Step 1: Load PDF & Chunk ---
def load_pdf(path, chunk_size=1500):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# --- Step 2: Embedding & FAISS Setup ---
chunks = load_pdf("sample.pdf")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# --- Step 3: Q&A with Ollama (Mistral) ---
def ask_mistral(question, context):
    prompt = f"""You are a helpful assistant. Answer the question using the following context.

Context:
{context}

Question: {question}
Answer:"""
    response = ollama.chat(model="mistral:7b", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# --- Step 4: Run Q&A Loop ---
print("✅ Mistral Q&A Bot Ready (type 'exit' to quit)")
while True:
    query = input("\nAsk about the PDF: ")
    if query.lower() == "exit":
        break

    q_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(q_embedding), k=3)
    context = "\n\n".join(chunks[i] for i in I[0])

    answer = ask_mistral(query, context)
    print("\n🧠 Answer:\n", answer.strip())
