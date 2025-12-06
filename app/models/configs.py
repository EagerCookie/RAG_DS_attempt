"""
Pydantic configuration models for all pipeline components
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


# ==========================================
# LOADER CONFIGURATIONS
# ==========================================

class LoaderConfigBase(BaseModel):
    type: str


class PDFLoaderConfig(LoaderConfigBase):
    type: Literal["pdf"] = "pdf"
    extract_images: bool = Field(default=True, description="Extract images from PDF")


class TextLoaderConfig(LoaderConfigBase):
    type: Literal["text"] = "text"
    encoding: str = Field(default="utf-8", description="Text file encoding")


LoaderConfig = PDFLoaderConfig | TextLoaderConfig


# ==========================================
# SPLITTER CONFIGURATIONS
# ==========================================

class SplitterConfigBase(BaseModel):
    type: str


class RecursiveSplitterConfig(SplitterConfigBase):
    type: Literal["recursive"] = "recursive"
    chunk_size: int = Field(default=1000, ge=100, le=4000, description="Size of each chunk")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Overlap between chunks")
    add_start_index: bool = Field(default=True, description="Add start index to metadata")


class SentenceTransformerSplitterConfig(SplitterConfigBase):
    type: Literal["sentence_transformer"] = "sentence_transformer"
    model_name: str = Field(default="DeepVk/USER-bge-m3", description="Model for tokenization")
    chunk_size: int = Field(default=256, ge=50, le=512, description="Token chunk size")
    chunk_overlap: int = Field(default=50, ge=0, le=256, description="Token overlap")


SplitterConfig = RecursiveSplitterConfig | SentenceTransformerSplitterConfig


# ==========================================
# EMBEDDING CONFIGURATIONS
# ==========================================

class EmbeddingConfigBase(BaseModel):
    type: str


class HuggingFaceEmbeddingConfig(EmbeddingConfigBase):
    type: Literal["huggingface"] = "huggingface"
    model_name: str = Field(default="DeepVk/USER-bge-m3", description="HuggingFace model name")
    device: Literal["cpu", "cuda"] = Field(default="cpu", description="Device to use")
    normalize_embeddings: bool = Field(default=True, description="Normalize embeddings")
    cache_folder: str = Field(default="./transformers_models", description="Cache directory")


EmbeddingConfig = HuggingFaceEmbeddingConfig


# ==========================================
# DATABASE CONFIGURATIONS
# ==========================================

class DatabaseConfigBase(BaseModel):
    type: str


class ChromaDBConfig(DatabaseConfigBase):
    type: Literal["chroma"] = "chroma"
    collection_name: str = Field(default="example_collection", description="Collection name")
    persist_directory: str = Field(default="./chroma_langchain_db", description="Storage directory")


class QdrantDBConfig(DatabaseConfigBase):
    type: Literal["qdrant"] = "qdrant"
    url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    collection_name: str = Field(default="my_documents", description="Collection name")
    api_key: Optional[str] = Field(default=None, description="API key for Qdrant Cloud")


DatabaseConfig = ChromaDBConfig | QdrantDBConfig


# ==========================================
# PIPELINE CONFIGURATION
# ==========================================

class PipelineConfig(BaseModel):
    name: str = Field(default="default_pipeline", description="Pipeline name")
    loader: LoaderConfig = Field(..., description="Loader configuration")
    splitter: SplitterConfig = Field(..., description="Splitter configuration")
    embedding: EmbeddingConfig = Field(..., description="Embedding configuration")
    database: DatabaseConfig = Field(..., description="Database configuration")


class PipelineConfigSimplified(BaseModel):
    """Упрощённая конфигурация пайплайна - только неизменяемое"""
    name: str = Field(default="default_pipeline")
    embedding: EmbeddingConfig = Field(..., description="Embedding configuration")
    database: DatabaseConfig = Field(..., description="Database configuration")


class ProcessingVariantConfig(BaseModel):
    """Вариант обработки - изменяемая часть"""
    name: str = "Default"
    loader: LoaderConfig
    splitter: SplitterConfig
    is_default: bool = False