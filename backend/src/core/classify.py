import logging
import os
import time
from typing import List

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from ..utils.debug import save_debug_image

logger = logging.getLogger("image_processing.classify")


# ------------------------------
# Device Config
# ------------------------------
def load_device() -> None:
    """Detect available devices and enable memory growth on GPUs."""
    devices = tf.config.list_physical_devices()
    gpus = tf.config.list_physical_devices("GPU")

    logger.info(f"[classify] Available devices: {devices}")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("✅ Enabled TensorFlow GPU memory growth")
        except RuntimeError as e:
            logger.exception(f"[classify] Failed to enable GPU memory growth: {e}")
    else:
        logger.warning("⚠️ No GPU detected — running on CPU.")


# ------------------------------
# Model Loading
# ------------------------------
def load_model(model_path: str):
    """Load and compile TensorFlow model."""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found: {model_path}")

        t0 = time.time()
        model = tf.keras.models.load_model(model_path)
        model.compile()
        elapsed = time.time() - t0
        logger.info(f"✅ Model loaded from {model_path} in {elapsed:.2f}s")
        return model
    except Exception as e:
        logger.exception(f"[classify] Failed to load model: {e}")
        raise


# ------------------------------
# Prediction Function
# ------------------------------
@tf.function(jit_compile=True)
def predict_fn(model, x):
    """Run compiled prediction (JIT-optimized)."""
    x = tf.cast(x, tf.float16)
    return model(x, training=False)


# ------------------------------
# Warm-up
# ------------------------------
def warm_up_model(model) -> None:
    """Run dummy inference to ensure model is ready."""
    try:
        t0 = time.time()
        dummy = tf.random.normal((1, 256, 256, 3))
        _ = predict_fn(model, dummy)
        elapsed = time.time() - t0
        logger.info(f"🔥 Model warm-up completed in {elapsed:.2f}s")
    except Exception as e:
        logger.exception(f"[classify] Warm-up failed: {e}")
        raise


# ------------------------------
# Preprocessing
# ------------------------------
def load_and_preprocess(img, target_size=(256, 256)):
    """Accepts a PIL image or NumPy array, resizes and preprocesses for inference."""
    try:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.resize(target_size)
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
        arr = preprocess_input(arr)        # normalize to [-1, 1]
        logger.debug(f"[classify] Preprocessed image -> shape={arr.shape}, dtype={arr.dtype}")
        return arr
    except Exception as e:
        logger.exception(f"[classify] Failed to preprocess image: {e}")
        raise


# ------------------------------
# Classification
# ------------------------------
def classify_masks(
    model,
    segmented_masks,
    threshold: float = 0.5
) -> List[Image.Image]:
    """
    Classifies masks and returns those above the threshold.

    Args:
        model: TensorFlow model used for prediction.
        segmented_masks: List of masks (PIL Images).
        threshold: Minimum score to keep a mask.

    Returns:
        List of masks with score >= threshold.
    """
    selected_images = []
    logger.info(f"[classify] Starting classification on {len(segmented_masks)} mask(s), threshold={threshold}")

    for idx, m in enumerate(segmented_masks, start=1):
        try:
            t0 = time.time()
            input_tensor = load_and_preprocess(m)
            pred = predict_fn(model, tf.convert_to_tensor(input_tensor))
            score = float(pred[0][0])
            elapsed = time.time() - t0

            logger.info(f"[classify] Mask {idx}: score={score:.4f}, time={elapsed:.3f}s")
            if score >= threshold:
                img_pil = Image.fromarray(m) if isinstance(m, np.ndarray) else m
                save_debug_image(img_pil, f"classified_pass_{idx}", prefix="cls")
                selected_images.append(m)
                logger.debug(f"[classify] Mask {idx} PASSED (score {score:.3f} ≥ {threshold})")
            else:
                logger.debug(f"[classify] Mask {idx} filtered out (score {score:.3f} < {threshold})")

        except Exception as e:
            logger.exception(f"[classify] Mask {idx} failed: {e}")
            continue

    logger.info(f"[classify] Completed: {len(selected_images)} / {len(segmented_masks)} mask(s) retained")
    return selected_images
