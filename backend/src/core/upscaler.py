from typing import List
import numpy as np
import onnxruntime
from PIL import Image


async def pre_process(in_image) -> np.ndarray:
    """
    Preprocesses image for ONNX model inference.
    H, W, C -> 1, C, H, W
    """
    arr = np.array(in_image.convert("RGB")).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # (1, 3, H, W)
    return arr


async def post_process(out_image) -> Image.Image:
    """
    Postprocesses ONNX model output back to image format.
    1, C, H, W -> H, W, C
    """
    arr = np.squeeze(out_image)
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


async def upscale_pil(selected_images: List[Image.Image], model_path: str = "backend/models/modelx4.ort") -> List[Image.Image]:
    sess = onnxruntime.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    upscaled_images = []
    for image in selected_images:
        inp = await pre_process(image)
        out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
        upscaled_images.append( await post_process(out))
    return upscaled_images