from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
import hashlib
import uuid
from datetime import datetime

# Import models
from app.models.configs import (
    PipelineConfig,
    LoaderConfig,
    PDFLoaderConfig,
    TextLoaderConfig,
    SplitterConfig,
    RecursiveSplitterConfig,
    SentenceTransformerSplitterConfig,
    EmbeddingConfig,
    HuggingFaceEmbeddingConfig,
    DatabaseConfig,
    ChromaDBConfig,
    QdrantDBConfig
)
from app.models.schemas import (
    ComponentInfo,
    PipelineResponse,
    ProcessingStatus
)

app = FastAPI(title="RAG Pipeline API", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import services AFTER app creation to avoid circular imports
from app.services.database import DatabaseManager
from app.services.pipeline_service import PipelineProcessor, PipelineValidator

# Initialize services
db_manager = DatabaseManager()

# In-memory cache for quick access (synced with DB)
pipelines_cache: Dict[str, PipelineConfig] = {}
tasks_cache: Dict[str, ProcessingStatus] = {}

# ==========================================
# REGISTRY & STORAGE
# ==========================================

class ComponentRegistry:
    """Registry for all available components"""
    
    LOADERS = {
        "pdf": {
            "name": "PDF Loader",
            "description": "Loads PDF files using PyPDFLoader",
            "config_class": PDFLoaderConfig,
            "file_extensions": [".pdf"]
        },
        "text": {
            "name": "Text Loader",
            "description": "Loads plain text files",
            "config_class": TextLoaderConfig,
            "file_extensions": [".txt", ".md"]
        }
    }
    
    SPLITTERS = {
        "recursive": {
            "name": "Recursive Character Splitter",
            "description": "Splits text recursively by character",
            "config_class": RecursiveSplitterConfig
        },
        "sentence_transformer": {
            "name": "Sentence Transformer Splitter",
            "description": "Splits by tokens using transformer model",
            "config_class": SentenceTransformerSplitterConfig
        }
    }
    
    EMBEDDINGS = {
        "huggingface": {
            "name": "HuggingFace Embeddings",
            "description": "Embeddings using HuggingFace models",
            "config_class": HuggingFaceEmbeddingConfig
        }
    }
    
    DATABASES = {
        "chroma": {
            "name": "Chroma DB",
            "description": "Local vector database",
            "config_class": ChromaDBConfig
        },
        "qdrant": {
            "name": "Qdrant",
            "description": "Qdrant vector database",
            "config_class": QdrantDBConfig
        }
    }

# Database manager (replace in-memory storage)
from app.services.database import DatabaseManager
from app.services.pipeline_service import PipelineProcessor, PipelineValidator

db_manager = DatabaseManager()

# In-memory cache for quick access (synced with DB)
pipelines_cache: Dict[str, PipelineConfig] = {}
tasks_cache: Dict[str, ProcessingStatus] = {}

# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
async def root():
    return {"message": "RAG Pipeline API", "version": "1.0.0"}

# Loaders
@app.get("/api/loaders", response_model=List[ComponentInfo])
async def list_loaders():
    """Get list of available loaders"""
    return [
        ComponentInfo(
            id=loader_id,
            name=info["name"],
            description=info["description"],
            config_schema=info["config_class"].model_json_schema()
        )
        for loader_id, info in ComponentRegistry.LOADERS.items()
    ]

@app.get("/api/loaders/{loader_id}", response_model=ComponentInfo)
async def get_loader(loader_id: str):
    """Get specific loader details"""
    if loader_id not in ComponentRegistry.LOADERS:
        raise HTTPException(status_code=404, detail="Loader not found")
    
    info = ComponentRegistry.LOADERS[loader_id]
    return ComponentInfo(
        id=loader_id,
        name=info["name"],
        description=info["description"],
        config_schema=info["config_class"].model_json_schema()
    )

# Splitters
@app.get("/api/splitters", response_model=List[ComponentInfo])
async def list_splitters():
    """Get list of available splitters"""
    return [
        ComponentInfo(
            id=splitter_id,
            name=info["name"],
            description=info["description"],
            config_schema=info["config_class"].model_json_schema()
        )
        for splitter_id, info in ComponentRegistry.SPLITTERS.items()
    ]

@app.get("/api/splitters/{splitter_id}", response_model=ComponentInfo)
async def get_splitter(splitter_id: str):
    """Get specific splitter details"""
    if splitter_id not in ComponentRegistry.SPLITTERS:
        raise HTTPException(status_code=404, detail="Splitter not found")
    
    info = ComponentRegistry.SPLITTERS[splitter_id]
    return ComponentInfo(
        id=splitter_id,
        name=info["name"],
        description=info["description"],
        config_schema=info["config_class"].model_json_schema()
    )

# Embeddings
@app.get("/api/embeddings", response_model=List[ComponentInfo])
async def list_embeddings():
    """Get list of available embedding models"""
    return [
        ComponentInfo(
            id=embed_id,
            name=info["name"],
            description=info["description"],
            config_schema=info["config_class"].model_json_schema()
        )
        for embed_id, info in ComponentRegistry.EMBEDDINGS.items()
    ]

# Databases
@app.get("/api/databases", response_model=List[ComponentInfo])
async def list_databases():
    """Get list of available vector databases"""
    return [
        ComponentInfo(
            id=db_id,
            name=info["name"],
            description=info["description"],
            config_schema=info["config_class"].model_json_schema()
        )
        for db_id, info in ComponentRegistry.DATABASES.items()
    ]

# Pipelines
@app.post("/api/pipelines", response_model=PipelineResponse)
async def create_pipeline(config: PipelineConfig):
    """Create a new pipeline configuration"""
    pipeline_id = str(uuid.uuid4())
    
    # Validate configuration
    warnings = PipelineValidator.validate_compatibility(config)
    
    # Save to database
    db_manager.save_pipeline(
        pipeline_id=pipeline_id,
        name=config.name,
        config=config.model_dump_json()
    )
    
    # Cache
    pipelines_cache[pipeline_id] = config
    
    response = PipelineResponse(
        pipeline_id=pipeline_id,
        config=config,
        created_at=datetime.now()
    )
    
    if warnings:
        response.warnings = warnings
    
    return response

@app.get("/api/pipelines/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(pipeline_id: str):
    """Get pipeline configuration"""
    # Try cache first
    if pipeline_id in pipelines_cache:
        config = pipelines_cache[pipeline_id]
    else:
        # Load from database
        pipeline_data = db_manager.get_pipeline(pipeline_id)
        if not pipeline_data:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        config = PipelineConfig.model_validate_json(pipeline_data['config'])
        pipelines_cache[pipeline_id] = config
    
    return PipelineResponse(
        pipeline_id=pipeline_id,
        config=config,
        created_at=datetime.now()
    )

@app.get("/api/pipelines")
async def list_pipelines():
    """List all pipelines"""
    pipelines = db_manager.list_pipelines()
    return pipelines

@app.post("/api/pipelines/{pipeline_id}/process")
async def process_file(
    pipeline_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Process a file using the specified pipeline"""
    # Load pipeline config
    pipeline_data = db_manager.get_pipeline(pipeline_id)
    if not pipeline_data:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    # Read file content
    content = await file.read()
    file_hash = hashlib.md5(content).hexdigest()
    
    # Check for duplicates
    existing = db_manager.file_exists(file_hash)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"File '{existing[0]}' already processed on {existing[1]}"
        )
    
    # Create task
    task_id = str(uuid.uuid4())
    db_manager.create_task(
        task_id=task_id,
        pipeline_id=pipeline_id,
        filename=file.filename,
        status="pending"
    )
    
    tasks_cache[task_id] = ProcessingStatus(
        task_id=task_id,
        status="pending",
        message="Task queued"
    )
    
    # Schedule background processing
    background_tasks.add_task(
        process_document_task,
        task_id,
        pipeline_id,
        file.filename,
        content,
        file_hash,
        len(content)
    )
    
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Processing started"
    }

@app.get("/api/tasks/{task_id}", response_model=ProcessingStatus)
async def get_task_status(task_id: str):
    """Get processing task status"""
    # Try cache first
    if task_id in tasks_cache:
        return tasks_cache[task_id]
    
    # Load from database
    task_data = db_manager.get_task(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = ProcessingStatus(
        task_id=task_data['id'],
        status=task_data['status'],
        progress=task_data['progress'],
        message=task_data['message'],
        error=task_data['error']
    )
    tasks_cache[task_id] = status
    
    return status

# Additional endpoints
@app.get("/api/files")
async def list_processed_files(limit: int = 100):
    """List processed files"""
    return db_manager.get_processed_files(limit=limit)

@app.get("/api/statistics")
async def get_statistics():
    """Get processing statistics"""
    return db_manager.get_statistics()

@app.delete("/api/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a pipeline"""
    success = db_manager.delete_pipeline(pipeline_id)
    if not success:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    # Clear from cache
    if pipeline_id in pipelines_cache:
        del pipelines_cache[pipeline_id]
    
    return {"message": "Pipeline deleted successfully"}

@app.post("/api/pipelines/{pipeline_id}/validate")
async def validate_pipeline(pipeline_id: str):
    """Validate pipeline configuration"""
    pipeline_data = db_manager.get_pipeline(pipeline_id)
    if not pipeline_data:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    config = PipelineConfig.model_validate_json(pipeline_data['config'])
    warnings = PipelineValidator.validate_compatibility(config)
    
    return {
        "valid": True,
        "warnings": warnings,
        "estimated_time_per_mb": PipelineValidator.estimate_processing_time(config, 1.0)
    }

# ==========================================
# BACKGROUND PROCESSING
# ==========================================

async def process_document_task(
    task_id: str,
    pipeline_id: str,
    filename: str,
    content: bytes,
    file_hash: str,
    file_size: int
):
    """Background task for document processing"""
    
    def update_progress(progress: float, message: str):
        """Helper to update task progress"""
        db_manager.update_task(
            task_id=task_id,
            progress=progress,
            message=message
        )
        if task_id in tasks_cache:
            tasks_cache[task_id].progress = progress
            tasks_cache[task_id].message = message
    
    try:
        # Update status to processing
        db_manager.update_task(task_id=task_id, status="processing")
        if task_id in tasks_cache:
            tasks_cache[task_id].status = "processing"
        
        # Load pipeline config
        pipeline_data = db_manager.get_pipeline(pipeline_id)
        config = PipelineConfig.model_validate_json(pipeline_data['config'])
        
        # Create processor
        processor = PipelineProcessor(config)
        
        # Process document
        results = await processor.process(
            content=content,
            filename=filename,
            progress_callback=update_progress
        )
        
        # Save file record
        db_manager.add_file(
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            pipeline_id=pipeline_id,
            chunks_count=results['chunks_created'],
            metadata=results
        )
        
        # Mark as completed
        db_manager.update_task(
            task_id=task_id,
            status="completed",
            progress=1.0,
            message=f"Successfully processed {filename}"
        )
        
        if task_id in tasks_cache:
            tasks_cache[task_id].status = "completed"
            tasks_cache[task_id].progress = 1.0
            tasks_cache[task_id].message = f"Successfully processed {filename}"
        
    except Exception as e:
        error_msg = str(e)
        db_manager.update_task(
            task_id=task_id,
            status="failed",
            error=error_msg,
            message="Processing failed"
        )
        
        if task_id in tasks_cache:
            tasks_cache[task_id].status = "failed"
            tasks_cache[task_id].error = error_msg
            tasks_cache[task_id].message = "Processing failed"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)