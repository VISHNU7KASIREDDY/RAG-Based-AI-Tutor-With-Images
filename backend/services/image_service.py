"""Embedding-based image retrieval utilizing cached JSON mapping and strict cosine math."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from services.embedding_service import _get_model

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METADATA_JSON = DATA_DIR / "image_metadata.json"
EMBEDDINGS_JSON = DATA_DIR / "image_embeddings.json"

_image_metadata: Optional[List[dict]] = None
_image_embeddings: Optional[List[dict]] = None

def _load_metadata() -> List[dict]:
    global _image_metadata
    if _image_metadata is None:
        if not METADATA_JSON.exists():
            return []
        with open(METADATA_JSON) as f:
            _image_metadata = json.load(f)
    return _image_metadata

def _load_or_generate_embeddings() -> List[dict]:
    global _image_embeddings
    if _image_embeddings is not None:
        return _image_embeddings

    metadata = _load_metadata()
    if not metadata:
        _image_embeddings = []
        return _image_embeddings

    if EMBEDDINGS_JSON.exists():
        with open(EMBEDDINGS_JSON) as f:
            _image_embeddings = json.load(f)
        return _image_embeddings

    _image_embeddings = []
    # Generate embeddings caching
    model = _get_model()
    for img in metadata:
        combined = img["title"] + " " + img["description"] + " " + " ".join(img.get("keywords", []))
        emb = model.encode([combined])[0]
        _image_embeddings.append({
            "id": img["id"],
            "embedding": emb.tolist()
        })
    with open(EMBEDDINGS_JSON, "w") as f:
        json.dump(_image_embeddings, f)
    
    return _image_embeddings

def get_embedding_by_id(img_id: str, image_embeddings: List[dict]) -> np.ndarray:
    for item in image_embeddings:
        if item["id"] == img_id:
            return np.array(item["embedding"])
    return np.zeros(384)

def get_relevant_image(query: str, image_metadata: List[dict], image_embeddings: List[dict], model) -> Optional[dict]:
    query_embedding = model.encode([query])[0]
    
    best_score = -1
    best_image = None
    
    for img in image_metadata:
        img_embedding = get_embedding_by_id(img["id"], image_embeddings)
        
        # cosine similarity
        score = np.dot(query_embedding, img_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(img_embedding)
        )
        
        if score > best_score:
            best_score = float(score)
            best_image = img
            
    # threshold check — don't return irrelevant image
    if best_score < 0.3:
        return None
        
    return best_image

def find_best_image(answer_text: str) -> Optional[dict]:
    """Wrapper function preserving backward compatibility mapping."""
    meta = _load_metadata()
    if not meta:
        return None
    embs = _load_or_generate_embeddings()
    model = _get_model()
    # Call core evaluation
    return get_relevant_image(answer_text, meta, embs, model)

def get_all_images() -> List[dict]:
    """Return all image metadata entries."""
    return _load_metadata()