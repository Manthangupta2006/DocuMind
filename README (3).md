# 📦 Text Chunker API

A FastAPI application that splits large text into chunks using LangChain's `RecursiveCharacterTextSplitter`, supports PDF upload, and allows LLM queries via Groq.

---

## 🚀 Features

| Feature | Endpoint | Description |
|---|---|---|
| Welcome | `GET /` | Returns a welcome message + available endpoints |
| Text Chunking | `POST /chunk` | Split large text into chunks |
| PDF Upload | `POST /upload-pdf` | Extract text from a PDF (optionally chunk it) |
| LLM Query | `POST /llm-query` | Ask any question to an LLM via Groq |

---

## 🗂️ Project Structure

```
.
├── main.py          # FastAPI app & route definitions
├── chunk.py         # Text chunking logic (RecursiveCharacterTextSplitter)
├── requirements.txt # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key (needed for `/llm-query`)
```bash
export GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at https://console.groq.com

### 4. Run the server
```bash
uvicorn main:app --reload
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## 📡 API Reference

### `GET /`
Returns a welcome message.

---

### `POST /chunk`
Split text into chunks.

**Request body:**
```json
{
  "text": "Your large text here...",
  "chunk_size": 50,
  "chunk_overlap": 10
}
```

**Response:**
```json
{
  "status": "success",
  "chunks": ["chunk1", "chunk2", "..."],
  "total_chunks": 5,
  "chunk_size_used": 50,
  "chunk_overlap_used": 10
}
```

---

### `POST /upload-pdf`
Upload a PDF file and extract its text.

**Query params:**
- `chunk_size` (default: 50)
- `chunk_overlap` (default: 10)
- `return_chunks` (default: false) — if true, also returns chunks of the extracted text

**Form data:** `file` — a `.pdf` file

---

### `POST /llm-query`
Ask a question to a Groq LLM.

**Request body:**
```json
{
  "query": "What is LangChain?",
  "model": "llama3-8b-8192"
}
```

**Response:**
```json
{
  "status": "success",
  "query": "What is LangChain?",
  "answer": "LangChain is a framework for ...",
  "model_used": "llama3-8b-8192",
  "usage": { "prompt_tokens": 30, "completion_tokens": 120, "total_tokens": 150 }
}
```

---

## ☁️ Deployment (Render)

1. Push your code to a public GitHub repo.
2. Go to [render.com](https://render.com) → **New Web Service**.
3. Connect your repo and set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port 10000`
4. Add `GROQ_API_KEY` as an environment variable.
5. Deploy!

---

## 🛡️ Input Validation & Error Handling

- `chunk_size` must be > 0
- `chunk_overlap` must be ≥ 0 and < `chunk_size`
- Text cannot be empty
- PDF must be `.pdf` extension and contain extractable text
- Groq errors return meaningful HTTP status codes (401, 400, 503)
