import base64
import io
import traceback

from PIL import Image
from fastapi import APIRouter, WebSocket, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse


from ..core import (
    image_preprocess,
    mask_image,
    classify_masks,
    upscale_pil,
    separate_color_layers,
    add_anchors_to_layers
)

router_image = APIRouter(
    prefix="/api/raw_image",
    tags=["image_processing"],
    default_response_class=JSONResponse,
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
    },
)


@router_image.post("/process")
async def process_image(
    request: Request,
    file: UploadFile = File(..., description="Image to be processed"),
):
    steps = [
        "load_image",
        "preprocess",
        "segment",
        "classify",
        "upscale",
        "color_separate",
        "add_anchors",
    ]

    try:
        current_step = steps[0]
        contents = await file.read()
        raw_image = Image.open(io.BytesIO(contents))

        # --- Step 1: preprocess ---
        current_step = steps[1]
        img = await image_preprocess(raw_image)

        # --- Step 2: segmentation ---
        current_step = steps[2]
        masks = await mask_image(img, request.app.state.GENERATOR)

        # --- Step 3: classification ---
        current_step = steps[3]
        classified = await classify_masks(request.app.state.MODEL, masks)

        # --- Step 4: upscaling ---
        current_step = steps[4]
        upscaled = await upscale_pil(classified)

        # --- Step 5: color layer separation ---
        current_step = steps[5]
        layers = await separate_color_layers(upscaled[0])

        # --- Step 6: anchor mark overlay ---
        current_step = steps[6]
        final = await add_anchors_to_layers(layers)

        encoded_layers = []
        for layer, color in final:
            buf = io.BytesIO()
            layer.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            encoded_layers.append({
                "color": color,
                "image_base64": encoded,
            })
        return {"status": "success", "layers": encoded_layers}
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        error_msg = f"❌ Error during '{current_step}' step: {type(e).__name__}: {str(e)}"
        print(error_msg)
        print(tb)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "step": current_step,
                "trace": tb.splitlines()[-3:],
            },
        )


@router_image.websocket("/ws/process")
async def ws_process(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"status": "connected", "progress": 0})

    try:
        # Receive image file bytes
        msg = await websocket.receive_bytes()
        raw_image = Image.open(io.BytesIO(msg))

        async def step(name, progress):
            """Utility to send progress updates."""
            await websocket.send_json({"stage": name, "progress": progress})

        # ---- Step 1: preprocess ----
        await step("preprocess", 10)
        img = await image_preprocess(raw_image)

        # ---- Step 2: segmentation ----
        await step("segment", 30)
        masks = await mask_image(img)

        # ---- Step 3: classification ----
        await step("classify", 50)
        classified = await classify_masks(MODEL, masks)

        # ---- Step 4: upscaling ----
        await step("upscale", 70)
        upscaled = await upscale_pil(classified)

        # ---- Step 5: color layer separation ----
        await step("color_layers", 90)
        layers = await separate_color_layers(upscaled[0])

        # ---- Step 6: add anchors ----
        final = await add_anchors_to_layers(layers)

        # Encode result (optional)
        await step("done", 100)
        await websocket.send_json({
            "status": "success",
            "layers": len(final)
        })

    except Exception as e:
        await websocket.send_json({"status": "error", "detail": str(e)})
    finally:
        await websocket.close()