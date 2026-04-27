# 📄 RAG PDF Q&A

A terminal-based Retrieval-Augmented Generation (RAG) system that lets you ask natural language questions about any PDF document. Built with LangChain, Google Gemini, and FAISS.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Hybrid Retrieval** | Combines semantic search (FAISS) with keyword search (BM25) for high-recall candidate retrieval |
| **Cross-Encoder Reranking** | Uses `ms-marco-MiniLM-L-6-v2` to re-score retrieved chunks by jointly encoding the query and document |
| **Semantic Chunking** | Splits PDFs at meaningful semantic boundaries rather than fixed character counts |
| **Streaming Output** | Answers are streamed token-by-token for an instant, typewriter-style experience |
| **Conversation Memory** | Maintains up to 6 prior turns so follow-up questions resolve correctly (e.g. "What did *he* say?") |
| **Smart Clarification** | Decision model asks for additional context only when the question is truly ambiguous |
| **Observability** | Traces every RAG run and clarification decision via [Langfuse](https://cloud.langfuse.com) |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Decision Model      │  ← Gemini Flash — decides if clarification is needed
│  (decide_next_step)  │
└────────┬────────────┘
         │ (clear question)
         ▼
┌─────────────────────┐
│  HybridRetriever     │
│  ├─ FAISS semantic   │  ← all-MiniLM-L6-v2 embeddings
│  └─ BM25 keyword     │  ← rank_bm25
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Cross-Encoder       │  ← ms-marco-MiniLM-L-6-v2
│  Reranker            │
└────────┬────────────┘
         │ (top-k reranked docs)
         ▼
┌─────────────────────┐
│  RAG Pipeline        │  ← Gemini Flash — generates streamed answer
│  (run_rag)           │    with conversation history injected into prompt
└─────────────────────┘
```

---

## 🚀 Quickstart

### 1. Clone & install dependencies

```bash
git clone https://github.com/<your-username>/RAG.git
cd RAG
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here
PDF_PATH=/path/to/your/document.pdf

# Langfuse observability (optional — leave blank to disable)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

> **Get a Google API key:** [Google AI Studio](https://aistudio.google.com/app/apikey)

### 3. Run

```bash
python main.py
```

---

## 💬 Usage

```
⟳  Loading embedding model...
⟳  Loading and chunking PDF (semantic)...
  ✓ Semantic chunking: 47 chunks
⟳  Building vector store...
⟳  Initialising retriever & cross-encoder...

==============================
   RAG PDF QA Terminal Ready!
==============================
  Type 'exit' to quit | 'history' to view conversation

────────────────────────────────────────────
Ask a question (or type 'exit'): What is this document about?

ANSWER:
The document is a report on advances in alloy development...

Sources:
  - /path/to/doc.pdf (page 1)
  - /path/to/doc.pdf (page 3)
```

### Special commands

| Command | Description |
|---|---|
| `exit` | Quit the program |
| `history` | Print the current conversation history |

---

## 📁 Project Structure

```
RAG/
├── main.py             # Entry point — init, main loop, conversation memory
├── config.py           # Env var loading & validation
├── load_data.py        # PDF loading + semantic/character chunking
├── embeddings_store.py # Embedding model & FAISS vector store
├── retriever.py        # Hybrid retrieval (FAISS + BM25) + cross-encoder reranking
├── rag_pipeline.py     # RAG chain with streaming & conversation history
├── decision_model.py   # Clarification decision via tool-calling
├── tools.py            # LangChain tool definition for clarification
├── observability.py    # Langfuse client + version-safe observe decorator
├── evaluation.py       # Recall@5 and MRR evaluation helpers
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── .env                # Local secrets (never committed)
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | **Required.** Google Generative AI API key |
| `PDF_PATH` | — | **Required.** Absolute path to your PDF |
| `CHUNK_SIZE` | `500` | Characters per chunk (fallback chunker only) |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks (fallback chunker only) |
| `SEMANTIC_K` | `5` | Number of docs from FAISS semantic search |
| `BM25_K` | `5` | Number of docs from BM25 keyword search |
| `RERANK_K` | `3` | Final top-k docs after cross-encoder reranking |
| `MAX_RETRIES` | `2` | Retry attempts on LLM/API failure |
| `RETRY_BACKOFF_SECONDS` | `1.0` | Base backoff between retries (exponential) |

---

## 🧠 Models Used

| Role | Model |
|---|---|
| LLM (answer generation) | `gemini-2.5-flash` |
| LLM (clarification decision) | `gemini-2.5-flash` |
| Embedding | `all-MiniLM-L6-v2` (HuggingFace) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

---
