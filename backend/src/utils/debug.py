from pathlib import Path
from PIL import Image
import logging
import uuid

logger = logging.getLogger("debug.save")

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # points to backend/
DEBUG_SAVE_ROOT = PROJECT_ROOT / "debug_results"
DEBUG_SAVE_ROOT.mkdir(parents=True, exist_ok=True)

def save_debug_image(image: Image.Image, stage: str, prefix: str = None) -> str:
    """
    Save intermediate image for debugging inside project/debug_results.
    Returns the saved file path.
    """
    try:
        image_id = prefix or str(uuid.uuid4())[:8]
        folder = DEBUG_SAVE_ROOT / image_id
        folder.mkdir(parents=True, exist_ok=True)

        path = folder / f"{stage}.png"
        image.save(path)
        logger.info(f"[debug.save] Saved {stage} → {path.relative_to(PROJECT_ROOT)}")
        return str(path)
    except Exception as e:
        logger.warning(f"[debug.save] Failed to save {stage}: {e}")
        return ""