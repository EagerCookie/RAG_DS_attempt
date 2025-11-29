"""
Pipeline Service - Handles actual document processing logic
"""
from typing import List, Any
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Import config models from models package
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


class LoaderFactory:
    """Factory for creating document loaders"""
    
    @staticmethod
    def create(config: LoaderConfig, filepath: str):
        if isinstance(config, PDFLoaderConfig):
            return PyPDFLoader(filepath)
        elif isinstance(config, TextLoaderConfig):
            # You can add TextLoader here
            from langchain_community.document_loaders import TextLoader
            return TextLoader(filepath, encoding=config.encoding)
        else:
            raise ValueError(f"Unknown loader type: {type(config)}")


class SplitterFactory:
    """Factory for creating text splitters"""
    
    @staticmethod
    def create(config: SplitterConfig):
        if isinstance(config, RecursiveSplitterConfig):
            return RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                add_start_index=config.add_start_index
            )
        elif isinstance(config, SentenceTransformerSplitterConfig):
            return SentenceTransformersTokenTextSplitter(
                model_name=config.model_name,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
        else:
            raise ValueError(f"Unknown splitter type: {type(config)}")


class EmbeddingFactory:
    """Factory for creating embedding models"""
    
    @staticmethod
    def create(config: EmbeddingConfig):
        if isinstance(config, HuggingFaceEmbeddingConfig):
            return HuggingFaceEmbeddings(
                model_name=config.model_name,
                model_kwargs={'device': config.device},
                encode_kwargs={'normalize_embeddings': config.normalize_embeddings},
                cache_folder=config.cache_folder
            )
        else:
            raise ValueError(f"Unknown embedding type: {type(config)}")


class DatabaseFactory:
    """Factory for creating vector database connections"""
    
    @staticmethod
    def save(config: DatabaseConfig, documents: List[Document], embeddings: Any):
        if isinstance(config, ChromaDBConfig):
            db = Chroma(
                collection_name=config.collection_name,
                embedding_function=embeddings,
                persist_directory=config.persist_directory,
            )
            return db.add_documents(documents)
        
        elif isinstance(config, QdrantDBConfig):
            # Uncomment when Qdrant is installed
            # from langchain_qdrant import Qdrant
            # return Qdrant.from_documents(
            #     documents,
            #     embeddings,
            #     url=config.url,
            #     collection_name=config.collection_name,
            #     api_key=config.api_key
            # )
            raise NotImplementedError("Qdrant support not yet implemented")
        else:
            raise ValueError(f"Unknown database type: {type(config)}")


class PipelineProcessor:
    """Main pipeline processor that orchestrates the entire flow"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    async def process(
        self, 
        content: bytes, 
        filename: str,
        progress_callback=None
    ) -> dict:
        """
        Process document through the entire pipeline
        
        Args:
            content: File content as bytes
            filename: Original filename
            progress_callback: Optional callback function(progress: float, message: str)
            
        Returns:
            dict with processing results
        """
        results = {
            "filename": filename,
            "chunks_created": 0,
            "embeddings_dimension": 0,
            "database_ids": []
        }
        
        # Save content to temporary file
        temp_file = None
        try:
            # 1. LOAD
            if progress_callback:
                progress_callback(0.1, "Loading document...")
            
            # Create temp file with proper extension
            suffix = os.path.splitext(filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                temp_file = tmp.name
            
            loader = LoaderFactory.create(self.config.loader, temp_file)
            documents = loader.load()
            
            results["documents_loaded"] = len(documents)
            
            if progress_callback:
                progress_callback(0.3, f"Loaded {len(documents)} documents")
            
            # 2. SPLIT
            if progress_callback:
                progress_callback(0.4, "Splitting documents...")
            
            splitter = SplitterFactory.create(self.config.splitter)
            chunks = splitter.split_documents(documents)
            
            results["chunks_created"] = len(chunks)
            
            if progress_callback:
                progress_callback(0.6, f"Created {len(chunks)} chunks")
            
            # 3. EMBEDDINGS
            if progress_callback:
                progress_callback(0.7, "Creating embeddings...")
            
            embeddings = EmbeddingFactory.create(self.config.embedding)
            
            # Test embedding dimension
            test_vec = embeddings.embed_query("test")
            results["embeddings_dimension"] = len(test_vec)
            
            if progress_callback:
                progress_callback(0.8, f"Embeddings ready (dim={len(test_vec)})")
            
            # 4. SAVE TO DATABASE
            if progress_callback:
                progress_callback(0.85, "Saving to vector database...")
            
            db_ids = DatabaseFactory.save(self.config.database, chunks, embeddings)
            results["database_ids"] = db_ids if db_ids else []
            
            if progress_callback:
                progress_callback(1.0, "Processing completed successfully")
            
            return results
            
        finally:
            # Cleanup temp file
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)


class PipelineValidator:
    """Validates pipeline configurations"""
    
    @staticmethod
    def validate_compatibility(config: PipelineConfig) -> List[str]:
        """
        Check if pipeline components are compatible
        
        Returns:
            List of warning messages (empty if all OK)
        """
        warnings = []
        
        # Example: Check if chunk size is appropriate for embedding model
        if isinstance(config.splitter, RecursiveSplitterConfig):
            if config.splitter.chunk_size > 2000:
                warnings.append(
                    "Large chunk size (>2000) may cause issues with some embedding models"
                )
        
        # Example: Check embedding model compatibility
        if isinstance(config.embedding, HuggingFaceEmbeddingConfig):
            if "bge" in config.embedding.model_name.lower():
                # BGE models work best with specific chunk sizes
                if isinstance(config.splitter, SentenceTransformerSplitterConfig):
                    if config.splitter.chunk_size > 512:
                        warnings.append(
                            "BGE models work best with chunk_size <= 512 tokens"
                        )
        
        return warnings
    
    @staticmethod
    def estimate_processing_time(config: PipelineConfig, file_size_mb: float) -> float:
        """
        Estimate processing time in seconds
        
        Args:
            config: Pipeline configuration
            file_size_mb: File size in megabytes
            
        Returns:
            Estimated time in seconds
        """
        # Very rough estimates - adjust based on your hardware
        base_time = file_size_mb * 2  # 2 seconds per MB base
        
        # Add time for embedding model loading
        if isinstance(config.embedding, HuggingFaceEmbeddingConfig):
            if "large" in config.embedding.model_name.lower():
                base_time += 30  # Large models take longer to load
            else:
                base_time += 10
        
        # Add time for chunking
        if isinstance(config.splitter, SentenceTransformerSplitterConfig):
            base_time *= 1.5  # Transformer splitters are slower
        
        return base_time