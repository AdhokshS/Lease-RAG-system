from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import numpy as np
import os

# -------------------------------
# Load embedding model
# -------------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# -------------------------------
# Read PDF
# -------------------------------
print("Reading PDF...")
reader = PdfReader("sample.pdf")

raw_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        raw_text += text + "\n"

print("Total characters extracted:", len(raw_text))

# -------------------------------
# Chunking
# -------------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

chunks = chunk_text(raw_text)

print("Number of chunks created:", len(chunks))

# -------------------------------
# Embed chunks
# -------------------------------
print("Embedding chunks...")
embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

# -------------------------------
# Create FAISS index
# -------------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

print("Adding chunk embeddings to FAISS...")
index.add(embeddings)

# -------------------------------
# Persist FAISS + chunks
# -------------------------------
os.makedirs("faiss_store", exist_ok=True)

faiss.write_index(index, "faiss_store/pdf_chunks.index")

with open("faiss_store/pdf_chunks.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk.replace("\n", " ") + "\n")

print("SUCCESS: PDF chunks embedded and stored in FAISS")
