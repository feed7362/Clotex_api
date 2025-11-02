import uuid
from pathlib import Path
import numpy as np
from PIL import Image
from transformers import pipeline
import torch


async def load_generator():
    """
    Loads the SAM2 model using the Hugging Face pipeline API.
    """
    try:
        generator = pipeline(
            task="mask-generation",
            model="facebook/sam2.1-hiera-tiny",
            device_map="auto",
            dtype=torch.float32,
        )
        torch.set_float32_matmul_precision("medium")
        print(f"✅ SAM2 generator loaded on: {generator.device}")
        return generator
    except Exception as e:
        print(f"❌ Failed to load SAM2 generator: {e}")
        return None


async def mask_image(raw_image: Image.Image, generator, size=(256, 256)) -> list[Image.Image]:
    """
    Generates segmented masked images from a raw image using the SAM2 mask-generation model.
    """
    if generator is None:
        raise ValueError("❌ Generator not initialized. Call load_generator() first.")

    # Use /tmp or module-local temp folder (safer in containers)
    image_output_dir = (Path(__file__).resolve().parent / "temp" / uuid.uuid4().hex)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    with torch.inference_mode():
        # with torch.autocast(device_type="cuda", enabled=False):
            outputs = generator(
                raw_image,
                points_per_batch=128,
                use_fast=True,
                multimask_output=False,
            )


    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    img_array = np.array(raw_image.convert("RGB"))
    segmented_masks = []

    for i, mask in enumerate(outputs["masks"]):
        mask = mask.squeeze()

        # Ensure tensor → NumPy conversion
        if isinstance(mask, torch.Tensor):
            mask = (mask > 0.5).to(torch.uint8).cpu().numpy()
        else:
            mask = (mask > 0.5).astype(np.uint8)


        # Resize and reapply mask
        mask_resized = Image.fromarray(mask * 255).resize(size=size, resample=Image.NEAREST)
        mask_array = np.array(mask_resized) // 255
        segmented_array = img_array * mask_array[..., None]

        segmented_img = Image.fromarray(segmented_array.astype(np.uint8))
        segmented_img.save(image_output_dir / f"segment_{i}.png")
        segmented_masks.append(segmented_img)

    print(f"✅ Created {len(segmented_masks)} segmented images in {image_output_dir}")
    return segmented_masks
