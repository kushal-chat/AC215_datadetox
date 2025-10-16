from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter(prefix="/test", tags=["test"])

class HealthCheckResponse(BaseModel):
    """
    Health check response
    """
    status: str = "I am healthy!"

@router.get(
    "/",
    summary="Health check endpoint",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheckResponse,
)
async def health_check() -> HealthCheckResponse:
    """
    Simple health check returning OK status.
    """
    return HealthCheckResponse()