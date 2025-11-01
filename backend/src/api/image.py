from fastapi import APIRouter
from fastapi.responses import JSONResponse

router_image = APIRouter(
    prefix="/api/image",
    tags=["image_processing"],
    default_response_class=JSONResponse,
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
    },
)

@router_image.get("/process")
async def process_image():
    return {"message": "Hello World"}