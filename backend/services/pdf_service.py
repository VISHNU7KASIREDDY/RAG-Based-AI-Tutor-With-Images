"""Extract and clean text from uploaded PDF files using PyMuPDF."""
import logging
import re
from typing import List
import fitz 
logger = logging.getLogger(__name__)
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
  """Open a PDF from raw bytes and return concatenated page text."""
  doc = fitz.open(stream=pdf_bytes, filetype="pdf")
  pages: List[str] = []
  for page_num, page in enumerate(doc):
    text = page.get_text()
    if text.strip():
      pages.append(text)
    logger.debug("Page %d: extracted %d chars", page_num + 1, len(text))
  doc.close()
  full_text = "\n\n".join(pages)
  return _clean_text(full_text)
def _clean_text(text: str) -> str:
  """Basic cleaning: collapse whitespace, strip odd control chars."""
  text = re.sub(r"[^\S\n]+", " ", text)    
  text = re.sub(r"\n{3,}", "\n\n", text)    
  text = re.sub(r"[^\x20-\x7E\n\r\t]", "", text) 
  return text.strip()