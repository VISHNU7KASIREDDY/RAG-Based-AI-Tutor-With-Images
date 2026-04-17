"""GET /images/{topicId} — Return image metadata catalogue."""
import logging
from fastapi import APIRouter
from fastapi.responses import FileResponse
from pathlib import Path
from services.image_service import get_all_images
logger = logging.getLogger(__name__)
router = APIRouter()
IMAGES_DIR = Path(__file__).resolve().parent.parent / "static" / "images"
@router.get("/images/{topic_id}")
async def list_images(topic_id: str):
  """Return all available image metadata."""
  images = get_all_images()
  return {"topicId": topic_id, "images": images}
@router.get("/images/file/{filename}")
async def serve_image(filename: str):
  """Serve an image file from the data/images directory."""
  file_path = IMAGES_DIR / filename
  if not file_path.exists():
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"Image '{filename}' not found")
  return FileResponse(str(file_path))