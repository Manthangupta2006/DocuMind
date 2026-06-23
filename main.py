from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import groq
import io
import os

from chunk import split_text_into_chunks

# ── Try importing PyPDF2 (optional) ──────────────────────────────────────────
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Text Chunker API",
    description=(
        "A FastAPI app that splits text into chunks using LangChain's "
        "RecursiveCharacterTextSplitter, supports PDF upload, and lets you "
        "query an LLM via Groq."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────
class ChunkRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text to split into chunks.")
    chunk_size: int = Field(50, gt=0, description="Max characters per chunk.")
    chunk_overlap: int = Field(10, ge=0, description="Overlapping characters between chunks.")


class LLMQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Your question for the LLM.")
    model: str = Field("llama3-8b-8192", description="Groq model to use.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def welcome():
    """Welcome endpoint."""
    return {
        "message": "👋 Welcome to the Text Chunker API!",
        "docs": "/docs",
        "endpoints": {
            "POST /chunk":      "Split text into chunks",
            "POST /upload-pdf": "Upload a PDF and get its text",
            "POST /llm-query":  "Ask a question to the LLM (Groq)",
        },
    }


@app.post("/chunk", tags=["Chunking"])
def chunk_text(request: ChunkRequest):
    """
    Split a large text into smaller chunks.

    - **text**: The input text (required).
    - **chunk_size**: Max characters per chunk (default 50).
    - **chunk_overlap**: Overlap between consecutive chunks (default 10).
    """
    try:
        result = split_text_into_chunks(
            text=request.text,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        return {"status": "success", **result}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/upload-pdf", tags=["PDF"])
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: int = Query(50, gt=0, description="Max characters per chunk."),
    chunk_overlap: int = Query(10, ge=0, description="Overlap between chunks."),
    return_chunks: bool = Query(False, description="Also chunk the extracted text?"),
):
    """
    Upload a PDF file → extract its text → optionally chunk it.

    Returns the raw extracted text (and chunks if `return_chunks=true`).
    """
    if not PDF_SUPPORT:
        raise HTTPException(
            status_code=501,
            detail="PyPDF2 is not installed. Run: pip install PyPDF2",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        contents = await file.read()
        reader = PyPDF2.PdfReader(io.BytesIO(contents))
        pages_text = [
            page.extract_text() or "" for page in reader.pages
        ]
        full_text = "\n".join(pages_text).strip()

        if not full_text:
            raise HTTPException(
                status_code=422,
                detail="Could not extract any text from the PDF (it may be scanned/image-based).",
            )

        response: dict = {
            "status": "success",
            "filename": file.filename,
            "total_pages": len(reader.pages),
            "extracted_text": full_text,
            "character_count": len(full_text),
        }

        if return_chunks:
            chunk_result = split_text_into_chunks(full_text, chunk_size, chunk_overlap)
            response["chunking"] = chunk_result

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.post("/llm-query", tags=["LLM"])
def llm_query(request: LLMQueryRequest):
    """
    Send a query to an LLM via the **Groq** API.

    Requires the `GROQ_API_KEY` environment variable to be set.

    Supported models (examples):
    - `llama3-8b-8192`
    - `llama3-70b-8192`
    - `mixtral-8x7b-32768`
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "GROQ_API_KEY environment variable is not set. "
                "Set it to use this endpoint."
            ),
        )

    try:
        client = groq.Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=request.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, concise assistant. "
                        "Answer the user's question clearly and accurately."
                    ),
                },
                {"role": "user", "content": request.query},
            ],
            max_tokens=1024,
            temperature=0.7,
        )

        answer = completion.choices[0].message.content
        return {
            "status": "success",
            "query": request.query,
            "answer": answer,
            "model_used": request.model,
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            },
        }

    except groq.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid GROQ_API_KEY.")
    except groq.BadRequestError as e:
        raise HTTPException(status_code=400, detail=f"Bad request to Groq: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
