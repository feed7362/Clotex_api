import logging
import time
import numpy as np
from PIL import Image
from transformers import pipeline
import torch
from ..utils.debug import save_debug_image

logger = logging.getLogger("image_processing.segment")


# ------------------------------
# Load SAM2 Generator
# ------------------------------
def load_generator():
    """
    Loads the SAM2 model using the Hugging Face pipeline API.
    """
    try:
        t0 = time.time()
        logger.info("[segment] Loading SAM2 generator (facebook/sam2.1-hiera-tiny)...")

        generator = pipeline(
            task="mask-generation",
            model="facebook/sam2.1-hiera-tiny",
            device=0,
            dtype="auto",
            use_fast=True,
            multimask_output=False,
        )
        generator.model = generator.model.to(torch.device("cuda"))
        torch.set_float32_matmul_precision("medium")

        elapsed = time.time() - t0
        device = getattr(generator, "device", "unknown")
        logger.info(f"✅ SAM2 generator loaded on: {device} in {elapsed:.2f}s")
        return generator
    except Exception as e:
        logger.exception(f"❌ Failed to load SAM2 generator: {e}")
        return None


# ------------------------------
# Run SAM2 Mask Generation
# ------------------------------
def mask_image(
    raw_image: Image.Image,
    generator
) -> list[Image.Image]:
    """
    Generates segmented masks from a raw image using the SAM2 mask-generation model.
    """
    if generator is None:
        raise ValueError("❌ Generator not initialized. Call load_generator() first.")

    try:
        logger.info(f"[segment] Starting mask generation — input size={raw_image.size}")
        t0 = time.time()

        with torch.inference_mode():
            outputs = generator(raw_image)

        # Ensure CUDA sync + cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.debug("[segment] GPU synchronized + cache cleared")

        segmented_masks = []

        masks = outputs.get("masks", [])
        logger.info(f"[segment] SAM2 returned {len(masks)} mask(s)")

        for i, mask in enumerate(masks, start=1):
            try:
                if isinstance(mask, torch.Tensor):
                    m = mask[0].cpu().numpy() if mask.ndim == 3 else mask.cpu().numpy()
                else:
                    m = np.asarray(mask)
                    if m.ndim == 3:
                        m = m[0]

                m = m.astype(np.float32)

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
