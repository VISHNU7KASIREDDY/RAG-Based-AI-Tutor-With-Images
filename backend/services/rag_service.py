"""Orchestrates the full RAG pipeline: embed query → retrieve → generate answer → find image."""
import logging
from typing import Dict, List, Optional
from services.embedding_service import generate_single_embedding, search_index
from services.llm_service import generate_answer
from services.image_service import find_best_image
logger = logging.getLogger(__name__)
TOP_K = 5
def ask(topic_id: str, query: str, debug: bool = False) -> dict:
  """
  Full RAG pipeline:
   1. Embed the user query
   2. Retrieve top-K chunks from FAISS
   3. Build context and call LLM
   4. Find the best matching image for the answer
   5. Return structured response
  """
  query_emb = generate_single_embedding(query)
  results = search_index(topic_id, query_emb, top_k=TOP_K)
  if not results:
    return {
      "answer": "No relevant information found in the document.",
      "image": None,
      "retrieved_chunks": [] if debug else None,
    }
  chunks = [chunk for chunk, _ in results]
  context = "\n\n---\n\n".join(chunks)
  answer = generate_answer(context, query)
  image_info = find_best_image(answer)
  response: Dict = {
    "answer": answer,
    "image": image_info,
  }
  if debug:
    response["retrieved_chunks"] = chunks
  logger.info("RAG pipeline complete for topic=%s query=%s", topic_id, query[:60])
  return response