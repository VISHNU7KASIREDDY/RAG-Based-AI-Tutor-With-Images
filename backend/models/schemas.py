"""Pydantic schemas for request / response models."""
from pydantic import BaseModel
from typing import Optional, List
class UploadResponse(BaseModel):
  topicId: str
  message: str = "PDF processed successfully"
  chunksCreated: int = 0
class ChatRequest(BaseModel):
  topicId: str
  query: str
class ImageInfo(BaseModel):
  filename: str
  title: str
  description: str
  similarity: Optional[float] = None
class ChatResponse(BaseModel):
  answer: str
  image: Optional[ImageInfo] = None
  sources: Optional[List[str]] = None
class ImageMetadata(BaseModel):
  id: str
  filename: str
  title: str
  keywords: List[str]
  description: str