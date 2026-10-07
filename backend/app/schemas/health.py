from datetime import datetime, timezone
from typing import List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """
    Schema for system operational health check.
    """
    status: str = Field(..., description="Operational status: healthy | degraded | unhealthy", json_schema_extra={"example": "healthy"})
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Server UTC timestamp")
    version: str = Field(..., description="Application version", json_schema_extra={"example": "1.0.0"})
    environment: str = Field(..., description="Runtime environment name", json_schema_extra={"example": "development"})


class SystemInfoResponse(BaseModel):
    """
    Schema for academic project metadata and non-diagnostic regulatory notice.
    """
    project_title: str = Field(..., description="Full academic project title")
    system_type: str = Field(..., description="System functional classification")
    disclaimer: str = Field(..., description="Mandatory non-diagnostic research disclaimer")
    version: str = Field(..., description="Application semantic version")
    modalities_supported: List[str] = Field(..., description="List of supported multimodal channels")
    baseline_paradigm: str = Field(..., description="Personalized baseline statistical approach")
