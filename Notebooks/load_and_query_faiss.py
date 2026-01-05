from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

print("Loading FAISS index from disk...")
index = faiss.read_index("faiss_store/rag.index")

print("Loading documents...")
with open("faiss_store/docs.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines()]

print("Index contains", index.ntotal, "vectors")

# -------------------------------
# Query
# -------------------------------
query = "How can we assess tenant credit default risk?"

print("\nQuery:", query)

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

k = 2
distances, indices = index.search(query_embedding, k)

print("\nTop matching documents:")
for idx in indices[0]:
    print("-", documents[idx])
