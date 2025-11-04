import asyncio
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.image import router_image
from src.api.health import router_health
from src.core.classify import load_device, load_model, warm_up_model
from src.core.upscaler import warmup_onnx_model
from src.core.mask_image import load_generator
from src.utils import clean_up_vram, setup_logging, clean_up_debug
import torch


# ------------------------------
# ASYNC LIFESPAN MANAGEMENT
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown lifecycle of the FastAPI application.
    Loads and warms up ML models, configures logging, and ensures cleanup.
    """
    t0 = time.time()
    setup_logging()
    logging.info("🧠 Initializing application lifespan context...")

    try:
        # --- Basic setup ---
        torch.set_float32_matmul_precision("medium")
        torch.set_grad_enabled(False)

        clean_up_debug()
        clean_up_vram()
        logging.info("✅ Cleaned up temporary GPU memory and cache")

        # ------------------------------
        # 🔹 Load all models in threads (so event loop not blocked)
        # ------------------------------
        logging.info("⚙️ Loading models in background threads...")

        # TensorFlow (sync functions wrapped)
        await asyncio.to_thread(load_device)
        model = await asyncio.to_thread(load_model, "./models/image_classifier.keras")
        await asyncio.to_thread(warm_up_model, model)
        app.state.MODEL = model
        logging.info("✅ TensorFlow model loaded & warmed up")

        # Torch SAM2 (sync)
        generator = await asyncio.to_thread(load_generator)
        if generator is None:
            logging.error("❌ Failed to initialize SAM2 generator")
        else:
            app.state.GENERATOR = generator
            logging.info("✅ SAM2 generator initialized")

        # ONNX upscaler (sync)
        upscaler = await asyncio.to_thread(warmup_onnx_model, "./models/modelx4.ort", (1, 3, 128, 128))
        app.state.UPSCALER = upscaler
        logging.info("✅ ONNX upscaler warmed up")

        total_time = time.time() - t0
        logging.info(f"🚀 Startup complete in {total_time:.2f}s — all models ready.")
        yield

    except Exception as e:
        logging.exception(f"❌ Fatal startup error: {e}")
        raise

    finally:
        try:
            clean_up_vram()
            logging.info("🧹 GPU and cache resources freed on shutdown")
        except Exception as e:
            logging.exception(f"⚠️ Cleanup error: {e}")
        logging.info("🛑 Shutdown complete.")


# ------------------------------
# APP FACTORY
# ------------------------------
def create_app(use_lifespan: bool = True) -> FastAPI:
    lifespan_ctx = lifespan if use_lifespan else None

    app = FastAPI(
        title="AI Automation Backend",
        description="Backend service powering the raw_image processing pipeline.",
        version="1.0.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan_ctx,
        swagger_ui_parameters={
            "deepLinking": True,
            "defaultModelsExpandDepth": 2,
            "displayRequestDuration": True,
            "docExpansion": "list",
        },
    )

    # Routers
    app.include_router(router_health)
    app.include_router(router_image)

    # CORS
    origins = ["http://localhost", "http://127.0.0.1"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logging.warning(f"HTTPException: {exc.detail} [{exc.status_code}]")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail or str(exc),
                "status_code": exc.status_code,
                "type": exc.__class__.__name__,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logging.exception(f"Unhandled error: {exc}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logging.warning(f"Validation error: {exc}")
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


# ------------------------------
# ENTRYPOINT
# ------------------------------
app = create_app()
