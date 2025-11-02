import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


async def image_preprocess(raw_image: Image.Image, size=(256, 256)) -> Image.Image:
    """
    Resize, denoise, enhance contrast, and normalize colors using only PIL + NumPy.
    """

    # 1️⃣ Ensure RGB mode
    if raw_image.mode != "RGB":
        raw_image = raw_image.convert("RGB")

    # 2️⃣ Resize early (faster processing)
    raw_image = raw_image.resize(size)

    # 3️⃣ Denoise — Gaussian blur
    raw_image = raw_image.filter(ImageFilter.GaussianBlur(radius=1.2))

    # 4️⃣ Enhance contrast (CLAHE-like but simpler)
    enhancer = ImageEnhance.Contrast(raw_image)
    raw_image = enhancer.enhance(1.4)  # 1.0 = no change, >1.0 = stronger contrast

    # 5️⃣ Normalize each channel via NumPy
    arr = np.asarray(raw_image).astype(np.float32)
    # Scale each channel to 0–255 range individually
    for c in range(3):
        ch = arr[..., c]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8) * 255.0
        arr[..., c] = ch
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # 6️⃣ Convert back to PIL Image
    preprocessed_image = Image.fromarray(arr)

    return preprocessed_image
