import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.image import router_image
from src.api.health import router_health
from src.core.classify import load_device, load_model, warm_up_model
from src.utils.memory import clean_up
from src.core.mask_image import load_generator

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    clean_up()
    logging.info("Cleaned vram gpu usage")

    await load_device()
    model = await load_model("./models/image_classifier.keras")
    await warm_up_model(model)
    app.state.MODEL = model
    logging.info("Warmuped tf model")

    generator = await load_generator()
    app.state.GENERATOR = generator  
    logging.info("Warmuped torch generator")
    
    logging.info("Startup complete. Initializing resources...")
    yield
    clean_up()
    logging.info("Free up gpu resources.")

    logging.info("Shutdown complete.")


def create_app(use_lifespan: bool = True) -> FastAPI:
    lifespan_ctx = lifespan if use_lifespan else None
    app = FastAPI(
        title="AI Automation Backend",
        description="Backend service powering the raw_image processing.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
        contact={
            "name": "John Doe",
            "email": "john@example.com",
        },
        license_info={
            "name": "MIT",
        },
        openapi_tags=[
            {
                "name": "image_processing",
                "description": "Endpoints for uploading, processing, and downloading video files.",
            },
            {
                "name": "health_check",
                "description": "Health check endpoints that provide liveness and readiness status.",
            },
        ],
        swagger_ui_parameters={
            "deepLinking": True,
            "defaultModelsExpandDepth": 2,  # show all models and schemas expanded
            "defaultModelExpandDepth": 2,  # expand individual model fields
            "displayRequestDuration": True,
            "displayOperationDuration": True,
            "defaultModelRendering": "example",
            "showMutatedRequest": True,
            "docExpansion": "list",  # expand all tags (groups) by default
            "supportedSubmitMethods": ["get", "post", "put", "delete", "patch"],
            "filter": True,
            "showExtensions": True,  # show any x-* vendor extensions
            "showCommonExtensions": True,  # show standard extensions like x-codeSamples
            "syntaxHighlight": True,  # enable syntax highlighting for request/response
            "requestSnippetsEnabled": True,
        },
        lifespan=lifespan_ctx,
    )

    app.include_router(router_health)
    app.include_router(router_image)


    origins = [
        "http://localhost",
        "http://127.0.0.1",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail or str(exc),
                "status_code": exc.status_code,
                "type": exc.__class__.__name__,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logging.exception(f"Unhandled error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)},
        )

    return app


app = create_app()
