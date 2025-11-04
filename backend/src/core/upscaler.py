import logging
import time
from typing import List
import numpy as np
import onnxruntime
from PIL import Image
from ..utils.debug import save_debug_image

logger = logging.getLogger("image_processing.upscale")


# ------------------------------
# Pre / Post Processing
# ------------------------------
def pre_process(in_image: Image.Image) -> np.ndarray:
    """
    Preprocess for ONNX inference.
    Converts RGB -> BGR and HWC -> NCHW.
    """
    # Convert PIL (RGB) → NumPy (BGR)
    arr = np.array(in_image.convert("RGB"))[:, :, ::-1]  # BGR
    arr = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
    logger.debug(f"[upscale] pre_process: shape={arr.shape}, dtype={arr.dtype}, range=({arr.min()}, {arr.max()})")
    return arr


def post_process(out_image: np.ndarray) -> Image.Image:
    """
    Postprocess ONNX output to PIL RGB.
    Converts NCHW -> HWC and BGR -> RGB.
    """
    arr = np.squeeze(out_image)
    arr = np.transpose(arr, (1, 2, 0))[:, :, ::-1]  # BGR→RGB
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    logger.debug(f"[upscale] post_process: shape={arr.shape}, dtype={arr.dtype}, range=({arr.min()}, {arr.max()})")
    return Image.fromarray(arr)


# ------------------------------
# Model Loading / Warmup
# ------------------------------
def warmup_onnx_model(model_path: str, input_shape: tuple):
    try:
        t0 = time.time()
        logger.info(f"[upscale] Loading ONNX model from: {model_path}")
        sess = onnxruntime.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        available = sess.get_providers()
        logger.info(f"[upscale] Available providers: {available}")

        input_name = sess.get_inputs()[0].name
        dummy = np.random.rand(*input_shape).astype(np.float32)
        _ = sess.run(None, {input_name: dummy})
        elapsed = time.time() - t0

        logger.info(f"[upscale] 🔥 Model warm-up complete in {elapsed:.2f}s ({available[0]})")
        return sess
    except Exception as e:
        logger.exception(f"[upscale] Failed to warm up ONNX model: {e}")
        raise


# ------------------------------
# Inference
# ------------------------------
def upscale_pil(selected_images: List[Image.Image], sess) -> List[Image.Image]:
    """
    Run ONNX-based upscaling for multiple images.
    """
    upscaled_images = []
    try:
        provider = sess.get_providers()[0] if sess else "Unknown"
        input_name = sess.get_inputs()[0].name
        logger.info(f"[upscale] Starting upscaling using {provider}")

        for idx, image in enumerate(selected_images, start=1):
            t0 = time.time()
            logger.debug(f"[upscale] Image {idx}: size={image.size}, mode={image.mode}")

            inp = pre_process(image)

            try:
                output = sess.run(None, {input_name: inp})[0]
            except Exception as e:
                logger.exception(f"[upscale] Inference failed on image {idx}: {e}")
                raise

            result_img = post_process(output)
            save_debug_image(result_img, f"upscaled_{idx}", prefix="up")
            upscaled_images.append(result_img)

            elapsed = time.time() - t0
            logger.info(f"[upscale] Image {idx}: done in {elapsed:.2f}s (input {image.size} → output {result_img.size})")

        logger.info(f"[upscale] Finished {len(upscaled_images)} image(s)")
        return upscaled_images

    except Exception as e:
        logger.exception(f"[upscale] Fatal error during batch upscaling: {e}")
        raise
