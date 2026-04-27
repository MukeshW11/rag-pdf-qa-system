# Riverr-RAG 📄🤖
### PDF Document Q&A System using RAG (Retrieval-Augmented Generation)

A production-style RAG pipeline that lets you ask natural language questions about any PDF document — powered by LangChain, HuggingFace embeddings, FAISS vector store, and a local Ollama LLM (no API keys required).

---

## Architecture

```
PDF File
   │
   ▼
PyPDFLoader ──► RecursiveTextSplitter ──► 500-token chunks
                                               │
                                               ▼
                              HuggingFace Embeddings (all-MiniLM-L6-v2)
                                               │
                                               ▼
                                      FAISS Vector Store
                                               │
                              ┌────────────────┘
                              │   Similarity Search (top-4 chunks)
                              ▼
                    Structured Prompt Template
                              │
                              ▼
                      Ollama LLM (llama3)
                              │
                              ▼
                     Answer + Source Citations
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS (CPU) |
| LLM | Ollama llama3 (local, free) |
| PDF Loader | PyPDF |
| Language | Python 3.10+ |

---

## Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/riverr-rag.git
cd riverr-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama & pull llama3
```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3
```

### 4. Run the Q&A system
```bash
python app.py documents/your_file.pdf
```

---

## Example Usage

```
📄 Loading PDF: documents/sample.pdf
   Loaded 12 pages
   Split into 87 chunks

🔍 Generating embeddings with HuggingFace...
   Vector store saved to 'faiss_index/'

🤖 Connecting to Ollama LLM (llama3)...

============================================================
  Riverr-RAG: PDF Q&A System
  Type your question | 'quit' to exit
============================================================

❓ Your Question: What is the main topic of this document?

⏳ Thinking...

💡 Answer:
The document discusses...

📎 Sources used:
   [1] Page 2: ...relevant chunk from the PDF...
   [2] Page 5: ...another relevant chunk...
```

---

## Key Features

- **No API keys needed** — runs 100% locally using Ollama
- **Hallucination control** — strict prompt template prevents made-up answers
- **Source citations** — every answer shows which pages were used
- **Persistent vector store** — FAISS index is saved locally, no re-embedding on restart
- **Chunk overlap** — 50-token overlap ensures context isn't lost at boundaries

---

## Project Structure

```
riverr-rag/
├── app.py              # Main RAG pipeline
├── requirements.txt    # Dependencies
├── README.md           # This file
├── faiss_index/        # Auto-generated vector store (gitignored)
└── documents/          # Place your PDFs here
```

---

## Future Enhancements

- [ ] Streamlit web UI for browser-based Q&A
- [ ] Multi-PDF support
- [ ] Conversation memory (multi-turn Q&A)
- [ ] RAGAS evaluation metrics
- [ ] Docker containerization

---

## Author

**Mukeshwaran A** — Full Stack AI Engineer  
[LinkedIn](https://www.linkedin.com/in/mukeshwaran-a-2236112a3) | [GitHub](https://github.com/mukeshwaran)
