from typing import Optional, Tuple

import numpy as np
from PIL import Image


async def add_anchors(
    image: Image.Image,
    cross_size: int = 10,
    margin: int = 15,
    color: Optional[Tuple[int, int, int]] = None,
    outline_width: int = 2,
) -> Image.Image:
    """
    Add registration anchor crosses (+) to all 4 corners of an image.

    Args:
        image: PIL Image to add anchors to
        cross_size: Half-length of each cross-arm (total cross = 2*cross_size + 1)
        margin: Distance from image edges to a center of cross
        color: RGB color tuple for crosses. If None, auto-detects contrasting color
        outline_width: Width of contrasting outline for better visibility

    Returns:
        PIL Image with anchor crosses added
    """
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to numpy array for efficient manipulation
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # Calculate corner positions
    positions = [
        (margin, margin),  # top-left
        (w - margin - 1, margin),  # top-right
        (margin, h - margin - 1),  # bottom-left
        (w - margin - 1, h - margin - 1),  # bottom-right
    ]

    # Draw crosses at each position
    for x, y in positions:
        # Skip if cross would go outside image bounds
        if (
            x < cross_size + outline_width
            or x >= w - cross_size - outline_width
            or y < cross_size + outline_width
            or y >= h - cross_size - outline_width
        ):
            continue

        # Auto-detect contrasting color if not specified
        if color is None:
            cross_color, outline_color = await _get_contrasting_colors(
                img_array, x, y, cross_size
            )
        else:
            cross_color = color
            brightness = np.mean(color)
            outline_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)

        # Draw cross with outline
        await _draw_cross(
            img_array, x, y, cross_size, cross_color, outline_color, outline_width
        )

    return Image.fromarray(img_array)


async def _get_contrasting_colors(
    img_array: np.ndarray, x: int, y: int, size: int
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Get contrasting cross and outline colors based on local image area."""
    # Sample area around the cross position
    y_start = max(0, y - size)
    y_end = min(img_array.shape[0], y + size + 1)
    x_start = max(0, x - size)
    x_end = min(img_array.shape[1], x + size + 1)

    local_area = img_array[y_start:y_end, x_start:x_end]
    avg_brightness = np.mean(local_area)

    # Choose contrasting colors
    if avg_brightness > 128:
        cross_color = (0, 0, 0)  # Black cross on bright background
        outline_color = (255, 255, 255)  # White outline
    else:
        cross_color = (255, 255, 255)  # White cross on dark background
        outline_color = (0, 0, 0)  # Black outline

    return cross_color, outline_color


async def _draw_cross(
    img_array: np.ndarray,
    x: int,
    y: int,
    size: int,
    cross_color: Tuple[int, int, int],
    outline_color: Tuple[int, int, int],
    outline_width: int,
) -> None:
    """Draw a cross with outline at specified position."""
    h, w = img_array.shape[:2]

    # Draw outline first (thicker)
    extended_size = size + outline_width
    x_start = max(0, x - extended_size)

    # Horizontal outline
    x_end = min(w, x + extended_size + 1)
    y_start = max(0, y - outline_width)
    y_end = min(h, y + outline_width + 1)
    img_array[y_start:y_end, x_start:x_end] = outline_color

    # Vertical outline
    x_end = min(w, x + outline_width + 1)
    y_start = max(0, y - extended_size)
    y_end = min(h, y + extended_size + 1)
    img_array[y_start:y_end, x_start:x_end] = outline_color

    # Draw main cross on top (thinner)
    # Horizontal line
    x_start = max(0, x - size)
    x_end = min(w, x + size + 1)
    img_array[y : y + 1, x_start:x_end] = cross_color

    # Vertical line
    y_start = max(0, y - size)
    y_end = min(h, y + size + 1)
    img_array[y_start:y_end, x : x + 1] = cross_color


async def add_anchors_to_layers(layers: list, **kwargs) -> list:
    """
    Add anchors to multiple layer images.

    Args:
        layers: List of (PIL Image, hex_color) tuples or just PIL Images
        **kwargs: Arguments to pass to add_anchors function

    Returns:
        List with anchors added to each layer (same format as input)
    """
    result = []

    for item in layers:
        if isinstance(item, tuple):
            # Handle (Image, hex_color) tuples
            image, hex_color = item
            anchored_image = await add_anchors(image, **kwargs)
            result.append((anchored_image, hex_color))
        else:
            # Handle just PIL Images
            anchored_image = await add_anchors(item, **kwargs)
            result.append(anchored_image)

    return result