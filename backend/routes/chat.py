"""POST /chat — RAG-based question answering with image retrieval."""
import logging
from fastapi import APIRouter, HTTPException, Query
from models.schemas import ChatRequest, ChatResponse, ImageInfo
from services.rag_service import ask
logger = logging.getLogger(__name__)
router = APIRouter()
@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, debug: bool = Query(False)):
  """
  Ask a question about a previously uploaded PDF.
  - Retrieves top-5 chunks from the FAISS index
  - Passes them as context to the LLM
  - Finds the best matching image for the answer
  - Returns answer + image metadata
  """
  if not body.query.strip():
    raise HTTPException(status_code=400, detail="Query must not be empty.")
  try:
    result = ask(topic_id=body.topicId, query=body.query, debug=debug)
    image = None
    if result.get("image"):
      img_data = result["image"].copy()
      img_data["filename"] = f"static/{img_data['filename']}"
      image = ImageInfo(**img_data)
    return ChatResponse(
      answer=result["answer"],
      image=image,
      sources=result.get("retrieved_chunks"),
    )
  except FileNotFoundError as exc:
    raise HTTPException(status_code=404, detail="Document not found. Please upload the PDF again.")
  except Exception as exc:
    logger.exception("Chat error")
    raise HTTPException(status_code=500, detail="Something went wrong while generating the answer.")