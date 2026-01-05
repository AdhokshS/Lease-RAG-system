from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np

# -------------------------------
# Load embedding model (retrieval)
# -------------------------------
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# -------------------------------
# Load FAISS index and chunks
# -------------------------------
print("Loading FAISS index and chunks...")
index = faiss.read_index("faiss_store/pdf_chunks.index")

with open("faiss_store/pdf_chunks.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f.readlines()]

# -------------------------------
# Load generation model
# -------------------------------
print("Loading generation model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# -------------------------------
# User question
# -------------------------------
question = "How can machine learning be used to predict tenant default risk?"

print("\nQuestion:", question)

# -------------------------------
# Retrieve relevant chunks
# -------------------------------
query_embedding = embed_model.encode([question])
query_embedding = np.array(query_embedding).astype("float32")

k = 3
_, indices = index.search(query_embedding, k)

retrieved_chunks = [chunks[i] for i in indices[0]]

context = " ".join(retrieved_chunks)

# -------------------------------
# Generate answer
# -------------------------------
prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{question}

Answer:
"""

inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
outputs = gen_model.generate(**inputs, max_new_tokens=150)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- Generated Answer ---\n")
print(answer)
