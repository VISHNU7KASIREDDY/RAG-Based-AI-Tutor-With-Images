"""Split cleaned text into overlapping word-level chunks."""
import logging
from typing import List
logger = logging.getLogger(__name__)
DEFAULT_CHUNK_SIZE = 300  
DEFAULT_OVERLAP = 50    
def chunk_text(
  text: str,
  chunk_size: int = DEFAULT_CHUNK_SIZE,
  overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
  """
  Split *text* into chunks of *chunk_size* words with *overlap* word overlap.
  Returns a list of text chunks.
  """
  words = text.split()
  if not words:
    return []
  chunks: List[str] = []
  start = 0
  while start < len(words):
    end = start + chunk_size
    chunk = " ".join(words[start:end])
    chunks.append(chunk)
    if end >= len(words):
      break
    start += chunk_size - overlap
  logger.info("Created %d chunks (size=%d, overlap=%d)", len(chunks), chunk_size, overlap)
  return chunks