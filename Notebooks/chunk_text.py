from pypdf import PdfReader

# Load PDF
reader = PdfReader("sample.pdf")

raw_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        raw_text += text + "\n"

print("Total characters in document:", len(raw_text))

# -------------------------------
# Chunking function
# -------------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # move back for overlap

    return chunks

# Create chunks
chunks = chunk_text(raw_text)

print("Number of chunks created:", len(chunks))

print("\n--- Sample Chunk ---\n")
print(chunks[0][:500])
