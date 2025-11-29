"""
API response schemas
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any, Literal
from datetime import datetime
from .configs import PipelineConfig


class ComponentInfo(BaseModel):
    """Information about a component (loader, splitter, etc.)"""
    id: str
    name: str
    description: str
    config_schema: Dict[str, Any]


class PipelineResponse(BaseModel):
    """Response when creating/getting a pipeline"""
    pipeline_id: str
    config: PipelineConfig
    created_at: datetime
    warnings: Optional[list[str]] = None


class ProcessingStatus(BaseModel):
    """Status of a processing task"""
    task_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    progress: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None