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
    
    # Generate vector DB identifier for this pipeline
    from app.services.pipeline_service import DatabaseFactory
    vector_db_identifier = DatabaseFactory.get_vector_db_identifier(
        config.database, 
        pipeline_id
    )
    
    # Save to database
    db_manager.save_pipeline(
        pipeline_id=pipeline_id,
        name=config.name,
        config=config.model_dump_json(),
        vector_db_identifier=vector_db_identifier
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
    
    # Check for duplicates IN THIS SPECIFIC PIPELINE
    existing = db_manager.file_exists_in_pipeline(file_hash, pipeline_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"File '{existing[0]}' already processed in this pipeline on {existing[1]}"
        )
    
    # Create task
    task_id = str(uuid.uuid4())
    db_manager.create_task(
        task_id=task_id,
        pipeline_id=pipeline_id,
        filename=file.filename,
        file_hash=file_hash,
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
    """List all processed files across all pipelines"""
    return db_manager.get_processed_files(limit=limit)

@app.get("/api/pipelines/{pipeline_id}/files")
async def list_pipeline_files(pipeline_id: str, limit: int = 100):
    """List files processed by specific pipeline"""
    pipeline_data = db_manager.get_pipeline(pipeline_id)
    if not pipeline_data:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    files = db_manager.get_files_by_pipeline(pipeline_id, limit=limit)
    return {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline_data['name'],
        "vector_db_identifier": pipeline_data['vector_db_identifier'],
        "files": files,
        "total_files": len(files)
    }

@app.get("/api/vector-databases")
async def list_vector_databases():
    """List all vector databases and their statistics"""
    pipelines = db_manager.list_pipelines()
    
    # Group by vector_db_identifier
    vector_dbs = {}
    for pipeline in pipelines:
        vdb_id = pipeline['vector_db_identifier']
        if vdb_id not in vector_dbs:
            stats = db_manager.get_vector_db_statistics(vdb_id)
            vector_dbs[vdb_id] = stats
    
    return {
        "vector_databases": list(vector_dbs.values()),
        "total_count": len(vector_dbs)
    }

@app.get("/api/vector-databases/{vector_db_identifier}/files")
async def list_vector_db_files(vector_db_identifier: str, limit: int = 100):
    """List all files in specific vector database"""
    files = db_manager.get_files_by_vector_db(vector_db_identifier, limit=limit)
    stats = db_manager.get_vector_db_statistics(vector_db_identifier)
    
    return {
        "vector_db_identifier": vector_db_identifier,
        "statistics": stats,
        "files": files,
        "total_files": len(files)
    }

@app.get("/api/files/{file_hash}/duplicates")
async def get_file_duplicates(file_hash: str):
    """Get all occurrences of a file across all pipelines"""
    duplicates = db_manager.get_file_duplicates(file_hash)
    
    return {
        "file_hash": file_hash,
        "occurrences": duplicates,
        "total_occurrences": len(duplicates),
        "unique_pipelines": len(set(d['pipeline_id'] for d in duplicates)),
        "unique_vector_dbs": len(set(d['vector_db_identifier'] for d in duplicates))
    }

@app.get("/api/statistics")
async def get_statistics():
    """Get processing statistics"""
    return db_manager.get_statistics()

@app.delete("/api/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a pipeline and all its files"""
    success = db_manager.delete_pipeline(pipeline_id)
    if not success:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    # Clear from cache
    if pipeline_id in pipelines_cache:
        del pipelines_cache[pipeline_id]
    
    return {"message": "Pipeline and all associated files deleted successfully"}

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
        vector_db_identifier = pipeline_data['vector_db_identifier']
        
        # Create processor with pipeline_id
        processor = PipelineProcessor(config, pipeline_id)
        
        # Process document
        results = await processor.process(
            content=content,
            filename=filename,
            progress_callback=update_progress
        )
        
        # Save file record with vector DB identifier
        db_manager.add_file(
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            pipeline_id=pipeline_id,
            vector_db_identifier=vector_db_identifier,
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