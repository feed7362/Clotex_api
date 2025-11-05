from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer, Group
from psd_tools.compression import Compression
import io
from typing import List


def psd_convertor(layer_images: List[Image.Image], layer_names: List[str]) -> bytes:
    """
    Create a layered PSD file using the modern psd-tools API.
    Each RGBA Pillow image becomes a toggleable PixelLayer.
    Returns PSD bytes ready to stream or save.
    """
    if not layer_images:
        raise ValueError("No layers provided.")

    width, height = layer_images[0].size
    psd = PSDImage.new(mode="RGB", size=(width, height), depth=8)

    for i, img in enumerate(layer_images):
        name = layer_names[i] if i < len(layer_names) else f"Layer_{i+1}"
        img_rgba = img.convert("RGBA")

        # Create a PixelLayer from a PIL image
        layer = PixelLayer.frompil(
            img_rgba,
            psd,
            name,
            top=0,
            left=0,
            compression=Compression.RLE,
        )
        psd.append(layer)

    group = Group.new("Color Layers", open_folder=True)
    group.extend(psd[:])
    psd.clear()
    psd.append(group)

    # Save PSD to bytes
    buf = io.BytesIO()
    psd.save(buf)
    return buf.getvalue()
