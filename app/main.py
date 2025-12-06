from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
import hashlib
import uuid
from datetime import datetime
import os

# ==========================================
# PROCESSING VARIANTS API
# ==========================================

from pydantic import BaseModel
from typing import Optional

# ВАЖНО: Загрузка переменных окружения из .env
from dotenv import load_dotenv
load_dotenv()

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
# MODELS
# ==========================================

class ProcessingVariantConfig(BaseModel):
    """Конфигурация варианта обработки"""
    name: str = "Default Variant"
    loader: LoaderConfig
    splitter: SplitterConfig
    description: Optional[str] = None


class CreateVariantRequest(BaseModel):
    """Запрос на создание варианта обработки"""
    pipeline_id: str
    variant_config: ProcessingVariantConfig


class ProcessWithVariantRequest(BaseModel):
    """Обработка файла с конкретным вариантом"""
    variant_id: str


class ProcessingVariantResponse(BaseModel):
    """Ответ с информацией о варианте"""
    variant_id: str
    pipeline_id: str
    name: str
    config: ProcessingVariantConfig
    created_at: str
    files_processed: int = 0


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

def process_document_task(
    task_id: str,
    pipeline_id: str,
    filename: str,
    content: bytes,
    file_hash: str,
    file_size: int
):
    """Background task for document processing (synchronous)"""
    import asyncio
    
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
        
        # Process document (run async function in sync context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(processor.process(
                content=content,
                filename=filename,
                progress_callback=update_progress
            ))
        finally:
            loop.close()
        
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
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        print(f"Error processing task {task_id}: {error_msg}")
        print(error_trace)
        
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

# ==========================================
# RAG INFERENCE ENDPOINTS
# Добавьте этот код в app/main.py
# ==========================================

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time

# Pydantic модели для RAG
class RAGQueryRequest(BaseModel):
    pipeline_id: str
    query: str
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    top_k: int = 5
    temperature: float = 0.0
    custom_api_url: Optional[str] = None


class RAGSource(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[RAGSource]
    metadata: Optional[Dict[str, Any]] = None


@app.post("/api/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Выполняет RAG запрос к базе знаний конкретного пайплайна
    """
    start_time = time.time()
    
    try:
        # 1. Загрузить конфигурацию пайплайна
        pipeline_data = db_manager.get_pipeline(request.pipeline_id)
        if not pipeline_data:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        config = PipelineConfig.model_validate_json(pipeline_data['config'])
        vector_db_identifier = pipeline_data['vector_db_identifier']
        
        # 2. Подключиться к векторной базе
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Использовать те же эмбеддинги, что и в пайплайне
        embedding_config = config.embedding
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_config.model_name,
            model_kwargs={'device': embedding_config.device},
            encode_kwargs={'normalize_embeddings': embedding_config.normalize_embeddings},
            cache_folder=embedding_config.cache_folder
        )
        
        # Подключиться к правильной коллекции
        db_config = config.database
        if isinstance(db_config, ChromaDBConfig):
            # Использовать полное имя коллекции с суффиксом pipeline_id
            collection_name = f"{db_config.collection_name}_{request.pipeline_id[:8]}"
            persist_directory = f"{db_config.persist_directory}/{request.pipeline_id[:8]}"
            
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory,
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Database type {db_config.type} not supported for RAG yet"
            )
        
        # 3. Поиск релевантных документов
        retrieved_docs = vector_store.similarity_search(
            request.query, 
            k=request.top_k
        )
        
        if not retrieved_docs:
            return RAGQueryResponse(
                answer="К сожалению, в базе знаний не найдено релевантной информации для ответа на ваш вопрос.",
                sources=[],
                metadata={
                    "retrieved_docs": 0,
                    "processing_time": f"{time.time() - start_time:.2f}s",
                    "llm_model": request.llm_model
                }
            )
        
        # 4. Создать контекст для LLM
        context = "\n\n".join([
            f"Документ {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # 5. Вызвать LLM
        llm_answer = await call_llm(
            provider=request.llm_provider,
            model=request.llm_model,
            query=request.query,
            context=context,
            temperature=request.temperature,
            custom_api_url=request.custom_api_url
        )
        
        # 6. Подготовить ответ
        sources = [
            RAGSource(
                content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in retrieved_docs
        ]
        
        processing_time = time.time() - start_time
        
        return RAGQueryResponse(
            answer=llm_answer,
            sources=sources,
            metadata={
                "retrieved_docs": len(retrieved_docs),
                "processing_time": f"{processing_time:.2f}s",
                "llm_model": request.llm_model,
                "llm_provider": request.llm_provider
            }
        )
        
    except Exception as e:
        import traceback
        print(f"RAG Query Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pipelines/{pipeline_id}/variants", response_model=ProcessingVariantResponse)
async def create_processing_variant(pipeline_id: str, variant_config: ProcessingVariantConfig):
    """
    Создать новый вариант обработки для существующего пайплайна
    
    Позволяет использовать разные loader/splitter с той же векторной БД
    """
    # Проверить, что пайплайн существует
    pipeline_data = db_manager.get_pipeline(pipeline_id)
    if not pipeline_data:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    # Проверить, что эмбеддинги совпадают
    base_config = PipelineConfig.model_validate_json(pipeline_data['config'])
    
    # Создать ID для варианта
    variant_id = str(uuid.uuid4())
    
    # Сохранить вариант
    db_manager.create_processing_variant(
        variant_id=variant_id,
        pipeline_id=pipeline_id,
        name=variant_config.name,
        config=variant_config.model_dump_json(),
        description=variant_config.description
    )
    
    return ProcessingVariantResponse(
        variant_id=variant_id,
        pipeline_id=pipeline_id,
        name=variant_config.name,
        config=variant_config,
        created_at=datetime.now().isoformat(),
        files_processed=0
    )


@app.get("/api/pipelines/{pipeline_id}/variants")
async def list_processing_variants(pipeline_id: str):
    """
    Получить все варианты обработки для пайплайна
    """
    pipeline_data = db_manager.get_pipeline(pipeline_id)
    if not pipeline_data:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    variants = db_manager.list_variants_for_pipeline(pipeline_id)
    
    return {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline_data['name'],
        "variants": variants,
        "total_variants": len(variants)
    }


@app.get("/api/variants/{variant_id}")
async def get_variant_details(variant_id: str):
    """
    Получить детали конкретного варианта
    """
    variant = db_manager.get_processing_variant(variant_id)
    if not variant:
        raise HTTPException(status_code=404, detail="Variant not found")
    
    config = ProcessingVariantConfig.model_validate_json(variant['config'])
    
    # Получить статистику по файлам
    files = db_manager.get_files_by_variant(variant_id)
    
    return {
        "variant_id": variant_id,
        "pipeline_id": variant['pipeline_id'],
        "name": variant['name'],
        "description": variant['description'],
        "config": config,
        "created_at": variant['created_at'],
        "files_processed": len(files),
        "files": files[:10]  # Первые 10
    }


@app.post("/api/variants/{variant_id}/process")
async def process_with_variant(
    variant_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Обработать файл используя конкретный вариант обработки
    
    Использует loader и splitter из варианта, но сохраняет в БД пайплайна
    """
    # Загрузить вариант
    variant = db_manager.get_processing_variant(variant_id)
    if not variant:
        raise HTTPException(status_code=404, detail="Variant not found")
    
    pipeline_id = variant['pipeline_id']
    
    # Загрузить базовый пайплайн (для эмбеддингов и БД)
    pipeline_data = db_manager.get_pipeline(pipeline_id)
    if not pipeline_data:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    # Прочитать файл
    content = await file.read()
    file_hash = hashlib.md5(content).hexdigest()
    
    # Проверка дубликатов (в рамках варианта + пайплайна)
    existing = db_manager.file_exists_in_pipeline(file_hash, pipeline_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"File '{existing[0]}' already processed in this pipeline"
        )
    
    # Создать задачу
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
    
    # Запустить обработку с вариантом
    background_tasks.add_task(
        process_document_with_variant_task,
        task_id,
        pipeline_id,
        variant_id,
        file.filename,
        content,
        file_hash,
        len(content)
    )
    
    return {
        "task_id": task_id,
        "variant_id": variant_id,
        "pipeline_id": pipeline_id,
        "status": "pending",
        "message": f"Processing with variant '{variant['name']}'"
    }


@app.delete("/api/variants/{variant_id}")
async def delete_variant(variant_id: str):
    """
    Удалить вариант обработки (файлы остаются в БД)
    """
    variant = db_manager.get_processing_variant(variant_id)
    if not variant:
        raise HTTPException(status_code=404, detail="Variant not found")
    
    db_manager.delete_processing_variant(variant_id)
    
    return {
        "message": "Variant deleted successfully",
        "note": "Files processed with this variant remain in the database"
    }

async def call_llm(
    provider: str,
    model: str,
    query: str,
    context: str,
    temperature: float,
    custom_api_url: Optional[str] = None
) -> str:
    """
    Вызывает LLM с контекстом из RAG
    """
    import os
    
    system_message = (
        "Ты — helpful AI assistant с доступом к базе знаний. "
        "Отвечай на вопросы пользователя, используя только информацию из предоставленного контекста. "
        "Если в контексте нет ответа, честно скажи об этом."
    )
    
    user_message = f"""Контекст из базы знаний:

{context}

---

Вопрос пользователя: {query}

Пожалуйста, ответь на вопрос, используя информацию из контекста выше."""
    
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise Exception("OPENAI_API_KEY not found in environment variables")
            
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key
            )
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            response = llm.invoke(messages)
            return response.content
        
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise Exception("ANTHROPIC_API_KEY not found in environment variables")
            
            llm = ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=api_key
            )
            
            messages = [
                {"role": "user", "content": f"{system_message}\n\n{user_message}"}
            ]
            
            response = llm.invoke(messages)
            return response.content
        
        elif provider == "deepseek":
            # Используйте OpenAI-совместимый API
            from langchain_openai import ChatOpenAI
            
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise Exception("DEEPSEEK_API_KEY not found in environment variables")
            
            llm = ChatOpenAI(
                base_url="https://api.deepseek.com/v1",
                api_key=api_key,
                model=model,
                temperature=temperature
            )
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            response = llm.invoke(messages)
            return response.content
        
        elif provider == "custom" and custom_api_url:
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(
                base_url=custom_api_url,
                api_key="dummy",  # Для локальных моделей
                model=model,
                temperature=temperature
            )
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            response = llm.invoke(messages)
            return response.content
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
    except Exception as e:
        raise Exception(f"LLM call failed: {str(e)}")


# Дополнительный эндпоинт для получения статистики по пайплайну
@app.get("/api/pipelines/{pipeline_id}/rag-info")
async def get_pipeline_rag_info(pipeline_id: str):
    """
    Получить информацию о пайплайне для RAG инференса
    """
    pipeline_data = db_manager.get_pipeline(pipeline_id)
    if not pipeline_data:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    config = PipelineConfig.model_validate_json(pipeline_data['config'])
    
    # Получить информацию о файлах
    files = db_manager.get_files_by_pipeline(pipeline_id, limit=1000)
    
    total_chunks = sum(f['chunks_count'] for f in files if f['chunks_count'])
    
    return {
        "pipeline_id": pipeline_id,
        "name": pipeline_data['name'],
        "vector_db_identifier": pipeline_data['vector_db_identifier'],
        "embedding_model": config.embedding.model_name,
        "embedding_dimension": None,  # Можно добавить при необходимости
        "total_files": len(files),
        "total_chunks": total_chunks,
        "database_type": config.database.type,
        "files": files[:10]  # Первые 10 для preview
    }


# ==========================================
# BACKGROUND TASK FOR VARIANT PROCESSING
# ==========================================

def process_document_with_variant_task(
    task_id: str,
    pipeline_id: str,
    variant_id: str,
    filename: str,
    content: bytes,
    file_hash: str,
    file_size: int
):
    """
    Обработка документа с использованием варианта
    
    Использует loader/splitter из варианта, но embedding/database из пайплайна
    """
    import asyncio
    
    def update_progress(progress: float, message: str):
        db_manager.update_task(task_id=task_id, progress=progress, message=message)
        if task_id in tasks_cache:
            tasks_cache[task_id].progress = progress
            tasks_cache[task_id].message = message
    
    try:
        db_manager.update_task(task_id=task_id, status="processing")
        if task_id in tasks_cache:
            tasks_cache[task_id].status = "processing"
        
        # Загрузить базовый пайплайн (для embedding/database)
        pipeline_data = db_manager.get_pipeline(pipeline_id)
        base_config = PipelineConfig.model_validate_json(pipeline_data['config'])
        vector_db_identifier = pipeline_data['vector_db_identifier']
        
        # Загрузить вариант (для loader/splitter)
        variant = db_manager.get_processing_variant(variant_id)
        variant_config = ProcessingVariantConfig.model_validate_json(variant['config'])
        
        # Создать гибридную конфигурацию
        hybrid_config = PipelineConfig(
            name=f"{base_config.name} - {variant_config.name}",
            loader=variant_config.loader,      # Из варианта!
            splitter=variant_config.splitter,  # Из варианта!
            embedding=base_config.embedding,   # Из пайплайна!
            database=base_config.database      # Из пайплайна!
        )
        
        # Создать процессор
        processor = PipelineProcessor(hybrid_config, pipeline_id)
        
        # Обработать документ
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(processor.process(
                content=content,
                filename=filename,
                progress_callback=update_progress
            ))
        finally:
            loop.close()
        
        # Сохранить с указанием варианта
        db_manager.add_file(
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            pipeline_id=pipeline_id,
            vector_db_identifier=vector_db_identifier,
            chunks_count=results['chunks_created'],
            metadata={
                **results,
                "variant_id": variant_id,
                "variant_name": variant['name']
            }
        )
        
        # Обновить таблицу files с variant_id
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE files SET variant_id = ? WHERE file_hash = ? AND pipeline_id = ?',
                (variant_id, file_hash, pipeline_id)
            )
        
        db_manager.update_task(
            task_id=task_id,
            status="completed",
            progress=1.0,
            message=f"Successfully processed with variant '{variant['name']}'"
        )
        
        if task_id in tasks_cache:
            tasks_cache[task_id].status = "completed"
            tasks_cache[task_id].progress = 1.0
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error processing with variant {variant_id}: {error_msg}")
        traceback.print_exc()
        
        db_manager.update_task(
            task_id=task_id,
            status="failed",
            error=error_msg,
            message="Processing failed"
        )
        
        if task_id in tasks_cache:
            tasks_cache[task_id].status = "failed"
            tasks_cache[task_id].error = error_msg

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)