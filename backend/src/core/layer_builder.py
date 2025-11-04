import logging
from typing import List, Tuple, Union
import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans, MiniBatchKMeans
from ..utils.debug import save_debug_image

logger = logging.getLogger("image_processing.color_layers")

# ------------------------------
# Auto-KMeans (Elbow Method)
# ------------------------------
def auto_kmeans(pixels, k_min=2, k_max=10):
    logger.debug(f"[auto_kmeans] evaluating K range {k_min}–{k_max}")
    inertia = []
    for k in range(k_min, k_max + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(pixels)
        inertia.append(km.inertia_)
        logger.debug(f"[auto_kmeans] k={k} inertia={km.inertia_:.2f}")

    deltas = np.diff(inertia)
    ratios = np.abs(deltas[1:] / (deltas[:-1] + 1e-9))
    k_auto = min(int(np.argmin(ratios) + k_min + 1), 10)  # +1 due to diff
    logger.info(f"[auto_kmeans] auto-selected K={k_auto}")
    return k_auto


def separate_color_layers(
    image: Image.Image,
    auto_k: int,
) -> List[Tuple[Image.Image, str]]:
    """
    Separates an image into color layers using k-means clustering in LAB space.
    Returns [(layer_img, hex_color), ...].
    """
    try:
        # --- Prepare image ---
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            img_rgb = np.array(image) / 255.0
            logger.debug(f"[color_layers] Converted PIL image {image.size}")
        else:
            img_rgb = image / 255.0
            logger.debug("[color_layers] Received raw array")

        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H,W,3), got {img_rgb.shape}")

        img_lab = rgb2lab(img_rgb)
        h, w, _ = img_lab.shape
        pixels = img_lab.reshape(-1, 3)
        logger.info(f"[color_layers] image {w}x{h}, total pixels={pixels.shape[0]}")

        # --- Determine cluster count ---
        if auto_k in (0, False, None):
            n_colors = auto_kmeans(pixels, 2, 15)
        elif isinstance(auto_k, int) and 1 <= auto_k <= 10:
            n_colors = auto_k
        else:
            raise ValueError(f"Invalid auto_k value: {auto_k}. Must be 0 for auto or 1–10.")

        logger.info(f"[color_layers] Using n_clusters={n_colors}")

        # --- Run KMeans ---
        kmeans = KMeans(
            n_clusters=n_colors,
            random_state=42,
            n_init=20,
            max_iter=500,
            init="k-means++",
        )
        labels = kmeans.fit_predict(pixels)
        cluster_centers_lab = kmeans.cluster_centers_
        labels = labels.reshape(h, w)
        logger.debug(f"[color_layers] KMeans fit complete, inertia={kmeans.inertia_:.2f}")

        # --- Convert centers to RGB ---
        cluster_centers_rgb = (lab2rgb(cluster_centers_lab.reshape(1, -1, 3))[0] * 255).astype(np.uint8)

        # --- Build color layers ---
        layers = []
        for i, color in enumerate(cluster_centers_rgb):
            mask = labels == i
            if not np.any(mask):
                logger.warning(f"[color_layers] Cluster {i} empty — skipped")
                continue

            layer_array = np.zeros((h, w, 3), dtype=np.uint8)
            layer_array[mask] = color
            layer_img = Image.fromarray(layer_array, mode="RGB")

            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            save_debug_image(layer_img, f"color_layer_{i}_{color_hex.strip('#')}", prefix="clr")
            layers.append((layer_img, color_hex, brightness))
            logger.debug(f"[color_layers] Cluster {i}: {color_hex}, brightness={brightness:.1f}, pixels={mask.sum()}")

        # --- Sort darkest→lightest ---
        layers.sort(key=lambda x: x[2])
        logger.info(f"[color_layers] Generated {len(layers)} color layers")

        return [(img, hex_color) for img, hex_color, _ in layers]

    except Exception as e:
        logger.exception(f"[color_layers] Failed to separate colors: {e}")
        raise


def separate_color_layers_batch(
    images: Union[Image.Image, List[Image.Image]],
    auto_k: int,
) -> List[Tuple[Image.Image, str]]:
    """
    Accepts a single image or list of images.
    Returns combined list of all color layers (image, color_hex).
    """
    try:
        if isinstance(images, (list, tuple)):
            results: List[Tuple[Image.Image, str]] = []
            logger.info(f"[color_layers_batch] Processing {len(images)} image(s)")
            for idx, img in enumerate(images, start=1):
                logger.info(f"[color_layers_batch] Image {idx} start")
                layers =  separate_color_layers(img, auto_k=auto_k)
                results.extend(layers)
                logger.info(f"[color_layers_batch] Image {idx} done, {len(layers)} layers")
            logger.info(f"[color_layers_batch] Total layers: {len(results)}")
            return results
        else:
            logger.info("[color_layers_batch] Single image input")
            return  separate_color_layers(images, auto_k=auto_k)

    except Exception as e:
        logger.exception(f"[color_layers_batch] Failed batch processing: {e}")
        raise
