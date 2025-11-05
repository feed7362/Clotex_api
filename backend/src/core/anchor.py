import logging
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
from ..utils.debug import save_debug_image

logger = logging.getLogger("image_processing.anchors")


def add_anchors(
    image: Image.Image,
    cross_size: int = 28,
    margin: int = 25,
    outline_width: int = 20,
) -> Image.Image:
    """Add strong registration anchors (+) to 4 corners."""
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_array = np.array(image)
        h, w = img_array.shape[:2]

        positions = [
            (margin, margin),  # top-left
            (w - margin, margin),  # top-right
            (margin, h - margin),  # bottom-left
            (w - margin, h - margin),  # bottom-right
        ]

        for (x, y) in positions:
            cross_color, outline_color = _get_contrasting_colors(img_array, x, y, cross_size)
            _draw_cross(img_array, x, y, cross_size, cross_color, outline_color, outline_width)

        result_img = Image.fromarray(img_array)
        save_debug_image(result_img, "anchored_preview", prefix="anc")
        return result_img

    except Exception as e:
        logger.exception(f"[Anchors] Failed to add anchors: {e}")
        raise


def _get_contrasting_colors(
    img_array: np.ndarray, x: int, y: int, size: int
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Get contrasting cross and outline colors based on local image area."""
    y_start = max(0, y - size)
    y_end = min(img_array.shape[0], y + size + 1)
    x_start = max(0, x - size)
    x_end = min(img_array.shape[1], x + size + 1)

    local_area = img_array[y_start:y_end, x_start:x_end]
    avg_brightness = np.mean(local_area)

    if avg_brightness > 128:
        cross_color = (0, 0, 0)
        outline_color = (255, 255, 255)
    else:
        cross_color = (255, 255, 255)
        outline_color = (0, 0, 0)

    logger.debug(
        f"[Anchors:_get_contrasting_colors] region=({x_start}:{x_end},{y_start}:{y_end}) "
        f"brightness={avg_brightness:.1f} cross={cross_color} outline={outline_color}"
    )
    return cross_color, outline_color


def _draw_cross(
    img_array: np.ndarray,
    x: int,
    y: int,
    size: int,
    cross_color: Tuple[int, int, int],
    outline_color: Tuple[int, int, int],
    outline_width: int,
) -> None:
    """Draw a cross with visible outline at (x, y)."""
    try:
        h, w = img_array.shape[:2]
        logger.debug(f"[Anchors:_draw_cross] ({x},{y}) start, size={size}, outline={outline_width}")

        # Ensure coordinates are within image
        x = np.clip(x, outline_width, w - outline_width - 1)
        y = np.clip(y, outline_width, h - outline_width - 1)

        # Draw outline (horizontal + vertical)
        for dy in range(-outline_width, outline_width + 1):
            img_array[np.clip(y + dy, 0, h - 1), max(0, x - size - outline_width):min(w, x + size + outline_width)] = outline_color
        for dx in range(-outline_width, outline_width + 1):
            img_array[max(0, y - size - outline_width):min(h, y + size + outline_width), np.clip(x + dx, 0, w - 1)] = outline_color

        # Draw main cross (solid color)
        img_array[y, max(0, x - size):min(w, x + size)] = cross_color
        img_array[max(0, y - size):min(h, y + size), x] = cross_color

    except Exception as e:
        logger.exception(f"[Anchors:_draw_cross] Failed at ({x},{y}): {e}")


def add_anchors_to_layers(layers: list, **kwargs) -> list[tuple[Image.Image, str]]:
    """
    Add anchors to multiple layers.
    Always returns [(Image, hex_color)].
    """
    result = []
    logger.info(f"[Anchors] Processing {len(layers)} layer(s)")

    for i, item in enumerate(layers, start=1):
        try:
            if isinstance(item, tuple):
                image, hex_color = item
            else:
                image, hex_color = item, f"{i:02x}{i:02x}{i:02x}"

            anchored = add_anchors(image, **kwargs)
            result.append((anchored, hex_color))
            logger.debug(f"[Anchors] Layer {i} done ({hex_color})")

        except Exception as e:
            logger.exception(f"[Anchors] Failed on layer {i}: {e}")
            continue

    logger.info(f"[Anchors] Finished {len(result)} layer(s)")
    return result
