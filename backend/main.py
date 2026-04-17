"""
RAG-Based AI Tutor — FastAPI application entry point.
Run with:
  cd backend && uvicorn main:app --reload
"""
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
  datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
app = FastAPI(
  title="RAG AI Tutor",
  description="Upload PDFs, ask questions, and get AI-generated answers with relevant images.",
  version="1.0.0",
)
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
from routes.upload import router as upload_router
from routes.chat import router as chat_router
from routes.images import router as images_router
app.include_router(upload_router, tags=["Upload"])
app.include_router(chat_router, tags=["Chat"])
app.include_router(images_router, tags=["Images"])
@app.get("/", tags=["Health"])
async def health():
  return {"status": "ok", "service": "RAG AI Tutor"}
@app.on_event("startup")
async def startup_event():
  logger.info(" RAG AI Tutor backend starting …")
  from services.embedding_service import _get_model
  _get_model()
  from services.image_service import _load_or_generate_embeddings
  _load_or_generate_embeddings()
  logger.info(" Startup complete")