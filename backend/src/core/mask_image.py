import logging
import time
import numpy as np
from PIL import Image
from ultralytics import FastSAM
import torch
from ..utils.debug import save_debug_image

logger = logging.getLogger("image_processing.segment")


# ------------------------------
# Load SAM2 Generator
# ------------------------------
def load_generator(model_path: str = "FastSAM-x.pt", device: str = "cuda"):
    """
    Loads the SAM2 model using the Hugging Face pipeline API.
    """
    try:
        t0 = time.time()
        logger.info(f"[FastSAM] Loading model from {model_path} on {device}...")

        model = FastSAM(model_path)
        model.to(device)
        torch.set_float32_matmul_precision("medium")

        # -------- Warm-up --------
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy)
        logger.info("[FastSAM] Warming up model on dummy image...")
        with torch.inference_mode():
            _ = model(dummy_image)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            logger.info(f"[FastSAM] CUDA warm-up complete, VRAM: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

        elapsed = time.time() - t0
        logger.info(f"✅ FastSAM loaded and warmed up in {elapsed:.2f}s")
        return model
    except Exception as e:
        logger.exception(f"[FastSAM] Failed to extract masks: {e}")
        return []


# ------------------------------
# Run SAM2 Mask Generation
# ------------------------------
def mask_image(
    raw_image: Image.Image,
    model
) -> list[Image.Image]:
    """
    Generates segmented masks from a raw image using the SAM2 mask-generation model.
    """
    if model is None:
        raise ValueError("❌ Generator not initialized. Call load_generator() first.")

    try:
        logger.info(f"[segment] Starting mask generation — input size={raw_image.size}")
        t0 = time.time()

        with torch.inference_mode():
            outputs = model(
                raw_image, 
                retina_masks=True, 
                imgsz=620, 
                conf=0.7, 
                iou=0.65,
                verbose=False,
                max_det=10
                )

        # Ensure CUDA sync + cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.debug("[segment] GPU synchronized + cache cleared")

        segmented_masks = []
        result = outputs[0] if isinstance(outputs, list) else outputs

        if not hasattr(result, "masks") or result.masks is None:
            logger.warning("[segment] No masks found in FastSAM output.")
            return []

        masks = result.masks.data
        
        logger.info(f"[segment] FastSAM returned {len(masks)} mask(s)")

        for i, mask in enumerate(masks, start=1):
            try:    
                m = mask.cpu().numpy().astype(np.float32)
                base = np.array(raw_image)
                segmented_img = (base * m[..., None]).astype(np.uint8)

                save_debug_image(Image.fromarray(segmented_img), f"mask_{i}", prefix="seg")
                segmented_masks.append(segmented_img)

                logger.debug(f"[segment] Mask {i}: applied successfully")

            except Exception as inner_e:
                logger.exception(f"[segment] Failed to process mask {i}: {inner_e}")
                raise
        
        elapsed = time.time() - t0
        logger.info(f"✅ Created {len(segmented_masks)} segmented images in {elapsed:.2f}s")

        return segmented_masks

    except torch.cuda.OutOfMemoryError as oom:
        logger.exception("❌ CUDA OOM during mask generation — consider smaller image or CPU fallback.")
        torch.cuda.empty_cache()
        raise oom

    except Exception as e:
        logger.exception(f"[segment] Unexpected error during segmentation: {e}")
        raise
