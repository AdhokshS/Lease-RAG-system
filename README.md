# AI-Powered Lease Document Q&A System (RAG)

This project is a lightweight, local **Retrieval-Augmented Generation (RAG)** system designed to answer questions from lease and real-estate-related documents **strictly using the content of the document**.

The goal of this project was to build a foundational understanding of:
- Embeddings
- Vector search
- Document chunking
- Retrieval vs generation
- Practical AI system design

This project was built as a learning-focused extension of an earlier **Tenant Credit Risk Analysis** system.

---

## 🔍 Problem Statement

Lease agreements and real estate documents are:
- Long
- Clause-heavy
- Difficult to query programmatically

Traditional keyword search fails to capture **semantic meaning**.

This system allows a user to:
1. Upload a lease document (PDF)
2. Ask natural-language questions
3. Receive answers grounded strictly in the document content

---

## 🧠 Architecture Overview

PDF Document
↓
Text Extraction
↓
Chunking (overlapping semantic blocks)
↓
Embeddings (Sentence-Transformers)
↓
Vector Storage (FAISS)
↓
Semantic Retrieval
↓
Context-Grounded Answer Generation

yaml
Copy code

---

## 🛠️ Tech Stack

- **Python 3.10**
- **Sentence-Transformers** – semantic embeddings
- **FAISS** – vector similarity search
- **PyTorch** – deep learning backend
- **Hugging Face Transformers** – local text generation
- **PyPDF** – PDF text extraction

All components run **locally**, with no external APIs.

---

## 🚀 Key Features

- Local, CPU-based embeddings
- Persistent vector store
- Chunk-level retrieval (not full documents)
- Reduced hallucination via document grounding
- Modular ingestion and query pipeline

---

## 📂 Project Scripts

| Script | Purpose |
|------|--------|
| `pdf_ingest.py` | Extract raw text from PDF |
| `chunk_text.py` | Create overlapping semantic chunks |
| `ingest_pdf_to_faiss.py` | Embed and store document chunks |
| `query_pdf_faiss.py` | Retrieve relevant chunks |
| `rag_query_with_generation.py` | Full RAG pipeline with generation |

---

## 📌 Learning Outcomes

- Clear separation of **retrieval** and **generation**
- Understanding of why chunk size and overlap matter
- Practical experience with vector databases
- Real-world dependency management
- Awareness of LLM hallucination risks and mitigation

---

## 🔮 Future Improvements

- Multi-document ingestion
- Metadata (page number, document source)
- Improved chunking strategies
- Stronger generation models
- UI layer (CLI or web app)

---

## 📬 About This Project

This project was built as part of a focused learning journey into AI and RAG systems, with an emphasis on **understanding fundamentals rather than using high-level frameworks**.
