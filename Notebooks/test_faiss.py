from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

documents = [
    "Tenant credit risk analysis evaluates the likelihood of rental default.",
    "Machine learning models can predict credit risk using historical data.",
    "ChromaDB is a vector database used for semantic search.",
    "Retrieval-Augmented Generation combines search with language models."
]

print("Creating embeddings...")
embeddings = model.encode(documents)

# FAISS requires float32
embeddings = np.array(embeddings).astype("float32")

print("Initializing FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

print("Adding vectors to FAISS...")
index.add(embeddings)

print("FAISS index contains", index.ntotal, "vectors")

# -------------------------------
# Persist FAISS + documents
# -------------------------------
print("Saving FAISS index and documents to disk...")

os.makedirs("faiss_store", exist_ok=True)

faiss.write_index(index, "faiss_store/rag.index")

with open("faiss_store/docs.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc + "\n")

print("SUCCESS: FAISS index and documents saved to disk")
