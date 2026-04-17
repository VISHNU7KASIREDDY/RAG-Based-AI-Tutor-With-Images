# RAG-Based AI Tutor with Image Retrieval

An intelligent tutoring system that uses **Retrieval-Augmented Generation (RAG)** to answer questions from uploaded PDF documents, and automatically retrieves relevant educational diagrams using **embedding-based image similarity**.

---

## Architecture

```
┌──────────────┐    ┌──────────────────────────────────────────────────┐
│  Streamlit │ HTTP │         FastAPI Backend         │
│  Frontend  │◄──────►│                         │
│       │    │ ┌────────┐ ┌──────────┐ ┌───────────────┐  │
│ • Upload  │    │ │ PDF  │→ │ Chunk  │→ │ Embedding  │  │
│ • Chat   │    │ │ Parser │ │ Service │ │ Service   │  │
│ • Images  │    │ └────────┘ └──────────┘ └───────┬───────┘  │
│       │    │                   │      │
│       │    │ ┌─────────────┐ ┌────────────────▼──────────┐ │
│       │    │ │ LLM Svc  │← │    FAISS Index    │ │
│       │    │ │ (Groq SDK) │ │ (per-topic vectors)   │ │
│       │    │ └──────┬──────┘ └──────────────────────────┘ │
│       │    │     │                    │
│       │    │ ┌──────▼──────┐ ┌──────────────────────────┐ │
│       │    │ │ RAG Svc  │→ │ Image Retrieval Svc   │ │
│       │    │ │ (orchestr.) │ │ (cosine similarity)   │ │
│       │    │ └─────────────┘ └──────────────────────────┘ │
└──────────────┘    └──────────────────────────────────────────────────┘
```

---

## RAG Pipeline

1. **Upload PDF** → Extract text with PyMuPDF → Clean & normalize
2. **Chunk** → Split into 300-word segments with 50-word overlap
3. **Embed** → Generate vectors using `all-MiniLM-L6-v2` (sentence-transformers)
4. **Store** → Save in a per-topic FAISS `IndexFlatL2` index with metadata
5. **Query** → Embed the question → Retrieve top-5 chunks → Pass to LLM as context
6. **Answer** → LLM generates a grounded answer using only the retrieved context

---

## Image Retrieval Logic

The system selects relevant educational diagrams **based on the LLM's answer**, not the raw query:

1. Pre-compute embeddings for each image's `title + description + keywords`
2. After the LLM generates an answer, embed the **full answer text**
3. Compute cosine similarity between the answer embedding and all image embeddings
4. Return the **single highest-scoring image**

This approach works well because the answer captures the semantic topic more precisely than the original question. If the similarity score is below 0.3, the system safely ignores displaying an image to avoid irrelevancy.

---

## Prompts Used

**LLM Tutor Grounding Prompt:**
```text
You are an AI tutor. Answer the student's question using ONLY the context provided below. If the answer is not in the context, say "I don't have enough information about that in this chapter."

Context:
{retrieved_chunks}

Student Question: {user_question}

Answer:
```

---

## Project Structure

```
├── backend/
│  ├── main.py          # FastAPI app entry point
│  ├── routes/
│  │  ├── upload.py       # POST /upload
│  │  ├── chat.py        # POST /chat
│  │  └── images.py       # GET /images/{topicId}
│  ├── services/
│  │  ├── pdf_service.py     # PDF text extraction
│  │  ├── chunk_service.py    # Text chunking
│  │  ├── embedding_service.py  # Embeddings + FAISS
│  │  ├── rag_service.py     # RAG orchestration
│  │  ├── image_service.py    # Image retrieval
│  │  └── llm_service.py     # Groq LLM
│  └── data/
│    ├── images.json      # Image catalogue
│    ├── images/        # Image files
│    └── faiss_index/      # Per-topic FAISS indices
├── frontend/
│  └── app.py           # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/your-username/RAG-Based-AI-Tutor-With-Images.git
cd RAG-Based-AI-Tutor-With-Images

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your API key:
#  GROQ_API_KEY=gsk_...
```

### 3. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Start the frontend

```bash
# In a new terminal
cd frontend
streamlit run app.py
```

The Streamlit app opens at **http://localhost:8501** and talks to the backend on port 8000.

---

## API Endpoints

### `POST /upload`

Upload a PDF for processing.

| Field | Type | Description |
|-------|------|-------------|
| `file` | `multipart/form-data` | PDF file |

**Response:**
```json
{
 "topicId": "a1b2c3d4e5f6",
 "message": "PDF processed successfully",
 "chunksCreated": 42
}
```

### `POST /chat`

Ask a question about an uploaded document.

**Request body:**
```json
{
 "topicId": "a1b2c3d4e5f6",
 "query": "How does a bell produce sound?"
}
```

**Query params:** `?debug=true` to include retrieved chunks.

**Response:**
```json
{
 "answer": "A bell produces sound through vibration...",
 "image": {
  "filename": "bell.png",
  "title": "Bell Vibration",
  "description": "Diagram showing how a bell vibrates to produce sound waves through mechanical oscillation",
  "similarity": 0.7823
 },
 "sources": null
}
```

### `GET /images/{topicId}`

List all available image metadata.

### `GET /static/images/{filename}`

Serve an image file directly.

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key |

**Default models:**
- Groq: `llama-3.1-8b-instant`

---

## Features

- PDF upload with automatic text extraction & chunking
- FAISS vector store with per-topic isolation
- RAG-based Q&A with strict context-only answers
- Embedding-based image retrieval (cosine similarity)
- Debug mode showing retrieved chunks
- Configurable LLM backend (Groq SDK natively configured)
- Clean service-based architecture
- Chat history in the frontend
- Responsive Streamlit UI with custom styling

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI |
| Frontend | Streamlit |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | FAISS (`IndexFlatL2`) |
| PDF Parsing | PyMuPDF |
| LLM | Groq SDK (`llama-3.1-8b-instant`) |

---

## License

MIT
