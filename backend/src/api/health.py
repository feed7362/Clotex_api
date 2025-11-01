from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..schemas.endpoint import HealthStatus

router_health = APIRouter(
    prefix="/api/health",
    tags=["health_check"],
    default_response_class=JSONResponse,
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
    },
)


@router_health.get(
    "/live",
    response_model=HealthStatus,
    summary="Liveness probe",
    description="Returns whether the API instance is running.",
    response_description="Current liveness status.",
    responses={
        200: {
            "model": HealthStatus,
            "description": "The service is up and responding.",
        }
    },
)
async def perform_liveness_checks() -> HealthStatus:
    return HealthStatus(status="ok")
