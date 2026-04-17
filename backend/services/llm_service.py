"""LLM integration using native Groq SDK."""
import logging
import os
from groq import Groq
logger = logging.getLogger(__name__)
SYSTEM_PROMPT = (
  "You are an AI tutor. Answer the student's question using "
  "ONLY the context provided below. If the answer is not in "
  "the context, say \"I don't have enough information about "
  "that in this chapter.\""
)
_client = None
def _get_client() -> Groq:
  global _client
  if _client is None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
      raise EnvironmentError("Missing API key. Set the GROQ_API_KEY environment variable.")
    _client = Groq(api_key=api_key)
    logger.info("LLM client initialised (provider=groq)")
  return _client
def generate_answer(context: str, question: str) -> str:
  """Send the RAG prompt to the Groq LLM and return the answer."""
  user_prompt = (
    f"Context:\n{context}\n\n"
    f"Student Question: {question}\n\n"
    "Answer:"
  )
  client = _get_client()
  model = "llama-3.1-8b-instant"
  logger.info("Generating answer with groq / %s", model)
  response = client.chat.completions.create(
    model=model,
    messages=[
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": user_prompt},
    ],
    temperature=0.3,
    max_tokens=1024,
  )
  return response.choices[0].message.content.strip()