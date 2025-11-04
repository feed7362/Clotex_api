import base64
import io
import json
import traceback
import zipfile
import uuid
import logging
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Request,
    WebSocket,
    Form,
)
from fastapi.responses import JSONResponse, FileResponse

from ..core import (
    mask_image,
    classify_masks,
    upscale_pil,
    separate_color_layers_batch,
    add_anchors_to_layers,
)

logger = logging.getLogger("image_processing")

router_image = APIRouter(
    prefix="/api/raw_image",
    tags=["image_processing"],
    default_response_class=JSONResponse,
    responses={404: {"description": "Not found"}, 500: {"description": "Internal server error"}},
)

TEMP_DIR = Path("/tmp/processed_images")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_INPUT_SIDE = 1400          # downscale very large inputs to save VRAM
THUMB_MAX_SIDE = 512           # shrink previews in JSON
WRITE_MANIFEST = True          # include manifest.json inside per-image folder


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _thumb_b64(img: Image.Image, max_side: int = THUMB_MAX_SIDE) -> str:
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return base64.b64encode(_png_bytes(img)).decode("utf-8")


@router_image.post("/process")
async def process_images(
    request: Request,
    files: List[UploadFile] = File(..., description="One or more images to process"),
    k_means: int = Form(0, description="0 = auto (K-means auto selection), 1–10 = manual number of colors"),
):
    """
    Processes multiple images. Returns:
    - base64 previews grouped by image (thumbnails to keep response light)
    - a ZIP archive with subfolders per image containing full-resolution layers + manifest.json
    """
    file_id = str(uuid.uuid4())
    zip_path = TEMP_DIR / f"{file_id}.zip"

    overall_results = []
    per_file_errors = []

    steps = ["load_image", "preprocess", "segment", "classify", "upscale", "color_separate", "add_anchors", "export_zip"]
    logger.info(f"[batch:{file_id}] start; {len(files)} file(s)")

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # process each file independently
            for file_index, upfile in enumerate(files, start=1):
                current_step = steps[0]
                file_name = getattr(upfile, "filename", f"image_{file_index}.png")
                name_stem = Path(file_name).stem or f"image_{file_index}"
                logger.info(f"[{file_name}] Step:{current_step} started")

                try:
                    # --- load & normalize image ---
                    contents = upfile.file.read() 
                    current_step = steps[1]
                    raw_image = Image.open(io.BytesIO(contents)).convert("RGB")
                    logger.info(f"[{file_name}] loaded {raw_image.size} mode={raw_image.mode}")

                    # --- 2. segment ---
                    current_step = steps[2]
                    logger.info(f"[{file_name}] Step:{current_step} started")
                    masks = mask_image(raw_image, request.app.state.GENERATOR)
                    logger.info(f"[{file_name}] Step:{current_step} ok")

                    # --- 3. classify ---
                    current_step = steps[3]
                    logger.info(f"[{file_name}] Step:{current_step} started")
                    classified = classify_masks(request.app.state.MODEL, masks)
                    logger.info(f"[{file_name}] Step:{current_step} ok")

                    # --- 4. upscale ---
                    current_step = steps[4]
                    logger.info(f"[{file_name}] Step:{current_step} started")
                    classified_pil = [
                        Image.fromarray(img) if isinstance(img, np.ndarray) else img
                        for img in classified
                    ]
                    upscaled = upscale_pil(classified_pil, request.app.state.UPSCALER)
                    logger.info(f"[{file_name}] Step:{current_step} ok")

                    # --- 5. color separation (handles single or list via *_batch) ---
                    current_step = steps[5]
                    logger.info(f"[{file_name}] Step:{current_step} started")
                    layers = separate_color_layers_batch(upscaled, k_means)
                    logger.info(f"[{file_name}] Step:{current_step} ok (layers={len(layers)})")

                    # --- 6. anchors ---
                    current_step = steps[6]
                    logger.info(f"[{file_name}] Step:{current_step} started")
                    final = add_anchors_to_layers(layers)
                    logger.info(f"[{file_name}] Step:{current_step} ok")

                    # --- 7. export (write PNGs + manifest) ---
                    current_step = steps[7]
                    encoded_layers = []
                    for idx, (layer_img, layer_hex) in enumerate(final, start=1):
                        rel_path = f"{name_stem}/layer_{idx}_{layer_hex}.png"
                        png_bytes = _png_bytes(layer_img)
                        zipf.writestr(rel_path, png_bytes)
                        encoded_layers.append({
                            "layer_number": idx,
                            "color_hex": layer_hex,
                            "path": rel_path,
                            "image_base64": _thumb_b64(layer_img),  # thumbnail for UI
                        })

                    if WRITE_MANIFEST:
                        manifest = {"filename": file_name, "layers": [{"layer_number": l["layer_number"], "color_hex": l["color_hex"], "path": l["path"]} for l in encoded_layers]}
                        zipf.writestr(f"{name_stem}/manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))

                    overall_results.append({"image": file_name, "layers": encoded_layers})
                    logger.info(f"[{file_name}] completed successfully")

                except Exception as e:
                    tb = traceback.format_exc(limit=5)
                    logger.exception(f"[{file_name}] failed at step '{current_step}': {e}")
                    per_file_errors.append({
                        "image": file_name,
                        "step": current_step,
                        "error": str(e),
                        "trace": tb.splitlines()[-4:],
                    })
                    # continue to next file

        if not overall_results:
            # all failed
            raise HTTPException(status_code=500, detail={"error": "All files failed to process", "errors": per_file_errors})

        return {
            "status": "partial_success" if per_file_errors else "success",
            "file_id": file_id,
            "download_url": f"/api/raw_image/download/{file_id}",
            "results": overall_results,
            "errors": per_file_errors,
        }

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc(limit=5)
        logger.exception(f"[batch:{file_id}] unexpected failure: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": tb.splitlines()[-4:]})


@router_image.get("/download/{file_id}")
def download_processed(file_id: str):
    zip_path = TEMP_DIR / f"{file_id}.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="File not found or expired")
    return FileResponse(path=zip_path, media_type="application/zip", filename=f"processed_layers_{file_id}.zip")


@router_image.websocket("/ws/process")
async def ws_process(websocket: WebSocket):
    """
    WebSocket version of /process endpoint.
    Processes one or multiple images with progressive stage updates.
    Sends messages like:
        {"stage": "segment", "progress": 40, "image": "sample.jpg"}
        {"status": "success", "file_id": "...", "results": [...]}
    """
    await websocket.accept()
    await websocket.send_json({"status": "connected", "progress": 0})

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())
    zip_path = TEMP_DIR / f"{file_id}.zip"

    overall_results = []
    per_file_errors = []

    steps = ["load_image", "segment", "classify", "upscale", "color_separate", "add_anchors", "export_zip"]

    try:
        # Expect a list of binary frames (multiple images)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            index = 0
            while True:
                try:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        logger.info("Client disconnected during processing")
                        break
                    if message["type"] != "websocket.receive_bytes":
                        continue

                    index += 1
                    contents = message["bytes"]
                    file_name = f"image_{index}.png"
                    name_stem = Path(file_name).stem

                    current_step = steps[0]
                    await websocket.send_json({"stage": current_step, "progress": 10, "image": file_name})
                    raw_image = Image.open(io.BytesIO(contents)).convert("RGB")
                    logger.info(f"[{file_name}] loaded {raw_image.size} mode={raw_image.mode}")

                    # --- 1. segment ---
                    current_step = steps[1]
                    await websocket.send_json({"stage": current_step, "progress": 30, "image": file_name})
                    masks = mask_image(raw_image, websocket.app.state.GENERATOR)

                    # --- 2. classify ---
                    current_step = steps[2]
                    await websocket.send_json({"stage": current_step, "progress": 50, "image": file_name})
                    classified = classify_masks(websocket.app.state.MODEL, masks)

                    # --- 3. upscale ---
                    current_step = steps[3]
                    await websocket.send_json({"stage": current_step, "progress": 70, "image": file_name})
                    classified_pil = [
                        Image.fromarray(img) if isinstance(img, np.ndarray) else img
                        for img in classified
                    ]
                    upscaled = upscale_pil(classified_pil, websocket.app.state.UPSCALER)

                    # --- 4. color separate ---
                    current_step = steps[4]
                    await websocket.send_json({"stage": current_step, "progress": 85, "image": file_name})
                    layers = separate_color_layers_batch(upscaled)

                    # --- 5. anchors ---
                    current_step = steps[5]
                    await websocket.send_json({"stage": current_step, "progress": 95, "image": file_name})
                    final = add_anchors_to_layers(layers)

                    # --- 6. export ---
                    current_step = steps[6]
                    encoded_layers = []
                    for idx, (layer_img, layer_hex) in enumerate(final, start=1):
                        rel_path = f"{name_stem}/layer_{idx}_{layer_hex}.png"
                        zipf.writestr(rel_path, _png_bytes(layer_img))
                        encoded_layers.append({
                            "layer_number": idx,
                            "color_hex": layer_hex,
                            "path": rel_path,
                            "image_base64": _thumb_b64(layer_img),
                        })

                    if WRITE_MANIFEST:
                        manifest = {
                            "filename": file_name,
                            "layers": [
                                {"layer_number": l["layer_number"], "color_hex": l["color_hex"], "path": l["path"]}
                                for l in encoded_layers
                            ],
                        }
                        zipf.writestr(
                            f"{name_stem}/manifest.json",
                            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
                        )

                    overall_results.append({"image": file_name, "layers": encoded_layers})
                    logger.info(f"[{file_name}] completed successfully")

                    await websocket.send_json(
                        {"stage": "completed", "progress": 100, "image": file_name}
                    )

                except Exception as e:
                    tb = traceback.format_exc(limit=5)
                    logger.exception(f"Error processing image {index}: {e}")
                    per_file_errors.append({
                        "image": f"image_{index}.png",
                        "step": current_step,
                        "error": str(e),
                        "trace": tb.splitlines()[-4:],
                    })
                    await websocket.send_json({
                        "status": "error",
                        "image": f"image_{index}.png",
                        "step": current_step,
                        "error": str(e),
                    })
                    continue  # allow next image

        # === Final summary ===
        if not overall_results:
            await websocket.send_json({
                "status": "failed",
                "error": "All files failed to process",
                "errors": per_file_errors,
            })
        else:
            await websocket.send_json({
                "status": "success",
                "file_id": file_id,
                "download_url": f"/api/raw_image/download/{file_id}",
                "results": overall_results,
                "errors": per_file_errors,
            })

    except Exception as e:
        tb = traceback.format_exc(limit=5)
        logger.exception(f"WebSocket processing failed: {e}")
        await websocket.send_json({
            "status": "fatal_error",
            "error": str(e),
            "trace": tb.splitlines()[-4:],
        })
    finally:
        await websocket.close()
