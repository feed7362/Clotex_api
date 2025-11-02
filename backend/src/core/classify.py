from typing import List
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# ---- Device Config ----
async def load_device() -> None:
    print("Available devices:", tf.config.list_physical_devices())

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Enabled TensorFlow memory growth")
        except RuntimeError as e:
            print(e)

# ---- Model Loading ----
async def load_model(model_path:str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    model.compile()  # optional: ensures it's ready for inference
    print("✅ Model loaded")
    return model

# ---- Prediction Function ----
@tf.function(jit_compile=True)
def predict_fn(model, x):
    """Run compiled prediction (JIT-optimized)"""
    return model(x, training=False)

# ---- Warmup ----
async def warm_up_model(model) -> None:
    dummy = tf.random.normal((1, 256, 256, 3))
    _ = predict_fn(model, dummy)
    print("🔥 Model warmed up")

# ---- Preprocessing ----
async def load_and_preprocess(img, target_size=(256, 256)):
    """Accepts a PIL image or NumPy array, resizes and preprocesses."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.resize(target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)  # shape: (1, H, W, 3)
    arr = preprocess_input(arr)        # normalize [-1,1]
    return arr

# ---- Classification ----
async def classify_masks(model, segmented_masks, threshold=0.5) -> List[Image.Image]:
    """
    Classifies masks and returns those above the given threshold.

    Args:
        model: TensorFlow model used for prediction.
        segmented_masks: List of masks (preprocessed or ready to preprocess).
        threshold: Minimum score to consider a mask as positive.

    Returns:
        List of tuples (mask, score) where score >= threshold.
    """
    selected_images = []

    for m in segmented_masks:
        input_tensor = await load_and_preprocess(m)
        pred = predict_fn(model, tf.convert_to_tensor(input_tensor))
        score = float(pred[0][0])

        if score >= threshold:
            selected_images.append(m)

    return selected_images