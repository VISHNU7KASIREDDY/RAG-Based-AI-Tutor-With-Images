"""Sentence-transformer embedding generation and FAISS index management."""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import faiss
import numpy as np
from fastembed import TextEmbedding
logger = logging.getLogger(__name__)
MODEL_NAME = "all-MiniLM-L6-v2"
_model: Optional[TextEmbedding] = None
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
def _get_model() -> TextEmbedding:
  """Lazy-load the embedding model."""
  global _model
  if _model is None:
    logger.info("Loading FastEmbed model: %s (threads=1)", MODEL_NAME)
    _model = TextEmbedding(f"sentence-transformers/{MODEL_NAME}", threads=1)
  return _model
def generate_embeddings(texts: List[str]) -> np.ndarray:
  """Return L2-normalised embeddings for a list of strings."""
  model = _get_model()
  embeddings = np.array(list(model.embed(texts))).astype("float32")
  faiss.normalize_L2(embeddings)
  return embeddings
def generate_single_embedding(text: str) -> np.ndarray:
  """Return a single normalised embedding vector (1-D)."""
  return generate_embeddings([text])[0]
def _index_dir(topic_id: str) -> Path:
  d = DATA_DIR / "faiss_index" / topic_id
  d.mkdir(parents=True, exist_ok=True)
  return d
def store_index(topic_id: str, embeddings: np.ndarray, chunks: List[str]) -> None:
  """Build a FAISS Inner-Product index and persist it alongside metadata."""
  dim = embeddings.shape[1]
  index = faiss.IndexFlatL2(dim)
  index.add(np.array(embeddings).astype("float32"))
  d = _index_dir(topic_id)
  faiss.write_index(index, str(d / "index.faiss"))
  meta = {"chunks": chunks}
  with open(d / "metadata.json", "w") as f:
    json.dump(meta, f)
  logger.info("Stored FAISS index for topic %s (%d vectors, dim=%d)", topic_id, len(chunks), dim)
def search_index(topic_id: str, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
  """Return the *top_k* most similar chunks with their scores."""
  d = _index_dir(topic_id)
  index_path = d / "index.faiss"
  meta_path = d / "metadata.json"
  if not index_path.exists():
    raise FileNotFoundError(f"No FAISS index found for topic {topic_id}")
  index = faiss.read_index(str(index_path))
  with open(meta_path) as f:
    meta = json.load(f)
  query_vec = np.expand_dims(query_embedding, axis=0).astype("float32")
  faiss.normalize_L2(query_vec)
  scores, indices = index.search(np.array(query_vec).astype("float32"), top_k)
  results: List[Tuple[str, float]] = []
  for score, idx in zip(scores[0], indices[0]):
    if idx < len(meta["chunks"]):
      results.append((meta["chunks"][idx], float(score)))
  return results