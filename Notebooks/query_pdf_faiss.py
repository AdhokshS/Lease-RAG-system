from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------------
# Load embedding model
# -------------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# -------------------------------
# Load FAISS index and chunks
# -------------------------------
print("Loading FAISS index...")
index = faiss.read_index("faiss_store/pdf_chunks.index")

print("Loading PDF chunks...")
with open("faiss_store/pdf_chunks.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f.readlines()]

print("Total chunks in index:", index.ntotal)

# -------------------------------
# User query
# -------------------------------
query = "How can machine learning be used to predict tenant default risk?"

print("\nQuery:", query)

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

# -------------------------------
# Search
# -------------------------------
k = 3
distances, indices = index.search(query_embedding, k)

print("\nTop relevant chunks:\n")
for rank, idx in enumerate(indices[0], start=1):
    print(f"--- Result {rank} ---")
    print(chunks[idx])
    print()
