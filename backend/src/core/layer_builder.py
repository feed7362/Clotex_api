from typing import List, Tuple

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans, MiniBatchKMeans


# ------------------------------
# Auto-KMeans (Elbow Method)
# ------------------------------
async def auto_kmeans(pixels, k_min=2, k_max=10):
    inertia = []
    for k in range(k_min, k_max + 1):
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5).fit(pixels)
        inertia.append(kmeans.inertia_)

    # Find "elbow" automatically (simple heuristic)
    deltas = np.diff(inertia)
    ratios = np.abs(deltas[1:] / deltas[:-1])
    k_auto = np.argmin(ratios) + k_min + 1  # +1 due to diff

    return k_auto


async def separate_color_layers(
    image: Image.Image, auto_k: bool | int = True
) -> List[Tuple[Image.Image, str]]:
    """
    Separates an image into color layers using k-means clustering in the LAB color space.
    The function identifies dominant colors in the image and segments it into corresponding
    layers, returning each layer along with its associated color in hexadecimal format.

    :param image: The input image, which can be a PIL Image instance or a NumPy array, and is
        expected to represent an RGB image.
    :param auto_k: Determines the number of color clusters. If True, an automatic method is
        used to decide the number of clusters. If an integer, it specifies the exact number
        of clusters. If False or invalid, an exception is raised.
    :return: A list of tuples, where each tuple contains a PIL Image representing a segmented
        color layer and its corresponding color in hexadecimal format.
    """
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        img_rgb = np.array(image) / 255.0  # normalize
    else:
        img_rgb = image / 255.0

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with shape (H, W, 3), got {img_rgb.shape}"
        )

    img_lab = rgb2lab(img_rgb)
    h, w, _ = img_lab.shape
    pixels = img_lab.reshape(-1, 3)

    # Determine K
    if isinstance(auto_k, bool) and auto_k:
        n_colors = await auto_kmeans(pixels, 2, 15)
    elif isinstance(auto_k, int):
        n_colors = auto_k
    else:
        raise ValueError("auto_k must be True, False, or an integer (e.g., 5)")

    kmeans = KMeans(
        n_clusters=n_colors, random_state=42, n_init=20, max_iter=500, init="k-means++"
    )
    labels = kmeans.fit_predict(pixels)
    cluster_centers_lab = kmeans.cluster_centers_
    labels = labels.reshape(h, w)

    # Convert cluster centers back to RGB
    cluster_centers_rgb = (
        lab2rgb(cluster_centers_lab.reshape(1, -1, 3))[0] * 255
    ).astype(np.uint8)

    # Create layers
    layers = []
    for i, color in enumerate(cluster_centers_rgb):
        # Create a mask for this cluster
        mask = labels == i

        # Skip empty layers
        if not np.any(mask):
            continue

        # Create a layer image with cluster color
        layer_array = np.zeros((h, w, 3), dtype=np.uint8)
        layer_array[mask] = color
        layer_image = Image.fromarray(layer_array, mode="RGB")

        # Convert color to hex
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

        # Calculate brightness for sorting
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

        layers.append((layer_image, color_hex, brightness))

    # Sort by brightness (darkest to lightest)
    layers.sort(key=lambda x: x[2])

    # Return image and hex color
    return [(img, hex_color) for img, hex_color, _ in layers]