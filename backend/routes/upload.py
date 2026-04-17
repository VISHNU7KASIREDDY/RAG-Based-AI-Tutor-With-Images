"""POST /upload — Accept a PDF, extract text, chunk, embed, and store in FAISS."""
import logging
import uuid
from fastapi import APIRouter, File, HTTPException, UploadFile
from models.schemas import UploadResponse
from services.pdf_service import extract_text_from_pdf
from services.chunk_service import chunk_text
from services.embedding_service import generate_embeddings, store_index
logger = logging.getLogger(__name__)
router = APIRouter()
@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
  """
  Upload a PDF document.
  Pipeline:
   1. Read PDF bytes
   2. Extract and clean text
   3. Chunk into ~400-word segments with 50-word overlap
   4. Generate embeddings
   5. Store FAISS index + metadata
   6. Return a unique topicId
  """
  if not file.filename or not file.filename.lower().endswith(".pdf"):
    raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
  try:
    pdf_bytes = await file.read()
    logger.info("Received PDF: %s (%d bytes)", file.filename, len(pdf_bytes))
    text = extract_text_from_pdf(pdf_bytes)
    if not text.strip():
      raise HTTPException(status_code=422, detail="Could not extract text from the PDF.")
    chunks = chunk_text(text)
    if not chunks:
      raise HTTPException(status_code=422, detail="No text chunks could be created.")
    embeddings = generate_embeddings(chunks)
    topic_id = uuid.uuid4().hex[:12]
    store_index(topic_id, embeddings, chunks)
    logger.info("PDF processed → topicId=%s, chunks=%d", topic_id, len(chunks))
    return UploadResponse(topicId=topic_id, chunksCreated=len(chunks), message="PDF processed successfully")
  except HTTPException:
    raise
  except Exception as exc:
    logger.exception("Error processing PDF")
    raise HTTPException(status_code=500, detail="Something went wrong during PDF processing. Please try again.")