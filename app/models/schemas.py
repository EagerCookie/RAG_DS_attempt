"""
API response schemas
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any, Literal, List
from datetime import datetime
from .configs import PipelineConfig, ProcessingVariantConfig


class ComponentInfo(BaseModel):
    """Information about a component (loader, splitter, etc.)"""
    id: str
    name: str
    description: str
    config_schema: Dict[str, Any]


class ProcessingVariantResponse(BaseModel):
    """Response for processing variant"""
    variant_id: str
    pipeline_id: str
    name: str
    description: Optional[str] = None
    config: ProcessingVariantConfig
    created_at: str
    files_processed: int = 0


class PipelineResponse(BaseModel):
    """Response when creating/getting a pipeline"""
    pipeline_id: str
    config: PipelineConfig
    created_at: datetime
    warnings: Optional[list[str]] = None
    variants: Optional[List[ProcessingVariantResponse]] = None  # Список вариантов
    default_variant_id: Optional[str] = None  # ID дефолтного варианта


class ProcessingStatus(BaseModel):
    """Status of a processing task"""
    task_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    progress: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None
    variant_id: Optional[str] = None  # Какой вариант использовался
