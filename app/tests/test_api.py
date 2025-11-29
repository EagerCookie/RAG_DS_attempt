"""
Test suite for RAG Pipeline API
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app, db_manager
import json
import io

client = TestClient(app)


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def sample_pipeline_config():
    """Sample pipeline configuration"""
    return {
        "name": "test_pipeline",
        "loader": {
            "type": "pdf",
            "extract_images": True
        },
        "splitter": {
            "type": "recursive",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "add_start_index": True
        },
        "embedding": {
            "type": "huggingface",
            "model_name": "DeepVk/USER-bge-m3",
            "device": "cpu",
            "normalize_embeddings": True
        },
        "database": {
            "type": "chroma",
            "collection_name": "test_collection",
            "persist_directory": "./test_data/chroma"
        }
    }


@pytest.fixture
def sample_pdf_file():
    """Create a mock PDF file"""
    # In real tests, you'd create a proper PDF
    # For now, return a bytes object
    return io.BytesIO(b"Mock PDF content")


@pytest.fixture(autouse=True)
def cleanup_db():
    """Cleanup database after each test"""
    yield
    # Clean up test data
    # db_manager.delete_all_test_data()


# ==========================================
# COMPONENT LISTING TESTS
# ==========================================

def test_list_loaders():
    """Test listing available loaders"""
    response = client.get("/api/loaders")
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) > 0
    
    # Check structure
    first_loader = data[0]
    assert "id" in first_loader
    assert "name" in first_loader
    assert "description" in first_loader
    assert "config_schema" in first_loader


def test_get_specific_loader():
    """Test getting specific loader details"""
    response = client.get("/api/loaders/pdf")
    assert response.status_code == 200
    data = response.json()
    
    assert data["id"] == "pdf"
    assert data["name"] == "PDF Loader"
    assert "config_schema" in data


def test_get_nonexistent_loader():
    """Test getting non-existent loader returns 404"""
    response = client.get("/api/loaders/nonexistent")
    assert response.status_code == 404


def test_list_splitters():
    """Test listing available splitters"""
    response = client.get("/api/splitters")
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) > 0


def test_list_embeddings():
    """Test listing available embedding models"""
    response = client.get("/api/embeddings")
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) > 0


def test_list_databases():
    """Test listing available vector databases"""
    response = client.get("/api/databases")
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) > 0


# ==========================================
# PIPELINE MANAGEMENT TESTS
# ==========================================

def test_create_pipeline(sample_pipeline_config):
    """Test creating a new pipeline"""
    response = client.post("/api/pipelines", json=sample_pipeline_config)
    assert response.status_code == 200
    data = response.json()
    
    assert "pipeline_id" in data
    assert "config" in data
    assert data["config"]["name"] == sample_pipeline_config["name"]


def test_create_pipeline_invalid_config():
    """Test creating pipeline with invalid config"""
    invalid_config = {
        "name": "test",
        "loader": {"type": "invalid_type"}
    }
    
    response = client.post("/api/pipelines", json=invalid_config)
    assert response.status_code == 422  # Validation error


def test_get_pipeline(sample_pipeline_config):
    """Test retrieving a pipeline"""
    # Create pipeline
    create_response = client.post("/api/pipelines", json=sample_pipeline_config)
    pipeline_id = create_response.json()["pipeline_id"]
    
    # Get pipeline
    get_response = client.get(f"/api/pipelines/{pipeline_id}")
    assert get_response.status_code == 200
    data = get_response.json()
    
    assert data["pipeline_id"] == pipeline_id
    assert data["config"]["name"] == sample_pipeline_config["name"]


def test_get_nonexistent_pipeline():
    """Test getting non-existent pipeline returns 404"""
    response = client.get("/api/pipelines/nonexistent-id")
    assert response.status_code == 404


def test_list_pipelines(sample_pipeline_config):
    """Test listing all pipelines"""
    # Create a pipeline
    client.post("/api/pipelines", json=sample_pipeline_config)
    
    # List pipelines
    response = client.get("/api/pipelines")
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) > 0


def test_delete_pipeline(sample_pipeline_config):
    """Test deleting a pipeline"""
    # Create pipeline
    create_response = client.post("/api/pipelines", json=sample_pipeline_config)
    pipeline_id = create_response.json()["pipeline_id"]
    
    # Delete pipeline
    delete_response = client.delete(f"/api/pipelines/{pipeline_id}")
    assert delete_response.status_code == 200
    
    # Verify deletion
    get_response = client.get(f"/api/pipelines/{pipeline_id}")
    assert get_response.status_code == 404


def test_validate_pipeline(sample_pipeline_config):
    """Test pipeline validation"""
    # Create pipeline
    create_response = client.post("/api/pipelines", json=sample_pipeline_config)
    pipeline_id = create_response.json()["pipeline_id"]
    
    # Validate pipeline
    validate_response = client.post(f"/api/pipelines/{pipeline_id}/validate")
    assert validate_response.status_code == 200
    data = validate_response.json()
    
    assert "valid" in data
    assert "warnings" in data
    assert data["valid"] is True


# ==========================================
# FILE PROCESSING TESTS
# ==========================================

def test_process_file(sample_pipeline_config, sample_pdf_file):
    """Test file processing"""
    # Create pipeline
    create_response = client.post("/api/pipelines", json=sample_pipeline_config)
    pipeline_id = create_response.json()["pipeline_id"]
    
    # Process file
    files = {"file": ("test.pdf", sample_pdf_file, "application/pdf")}
    process_response = client.post(
        f"/api/pipelines/{pipeline_id}/process",
        files=files
    )
    
    assert process_response.status_code == 200
    data = process_response.json()
    
    assert "task_id" in data
    assert "status" in data
    assert data["status"] == "pending"


def test_process_file_nonexistent_pipeline(sample_pdf_file):
    """Test processing file with non-existent pipeline"""
    files = {"file": ("test.pdf", sample_pdf_file, "application/pdf")}
    response = client.post(
        "/api/pipelines/nonexistent-id/process",
        files=files
    )
    
    assert response.status_code == 404


def test_process_file_without_file(sample_pipeline_config):
    """Test processing without providing a file"""
    create_response = client.post("/api/pipelines", json=sample_pipeline_config)
    pipeline_id = create_response.json()["pipeline_id"]
    
    response = client.post(f"/api/pipelines/{pipeline_id}/process")
    assert response.status_code == 422  # Validation error


def test_process_duplicate_file(sample_pipeline_config, sample_pdf_file):
    """Test processing same file twice (should fail)"""
    # Create pipeline
    create_response = client.post("/api/pipelines", json=sample_pipeline_config)
    pipeline_id = create_response.json()["pipeline_id"]
    
    # Process file first time
    files = {"file": ("test.pdf", sample_pdf_file, "application/pdf")}
    first_response = client.post(
        f"/api/pipelines/{pipeline_id}/process",
        files=files
    )
    assert first_response.status_code == 200
    
    # Reset file pointer
    sample_pdf_file.seek(0)
    
    # Try to process same file again
    files = {"file": ("test.pdf", sample_pdf_file, "application/pdf")}
    second_response = client.post(
        f"/api/pipelines/{pipeline_id}/process",
        files=files
    )
    assert second_response.status_code == 409  # Conflict


# ==========================================
# TASK STATUS TESTS
# ==========================================

def test_get_task_status(sample_pipeline_config, sample_pdf_file):
    """Test getting task status"""
    # Create pipeline and process file
    create_response = client.post("/api/pipelines", json=sample_pipeline_config)
    pipeline_id = create_response.json()["pipeline_id"]
    
    files = {"file": ("test.pdf", sample_pdf_file, "application/pdf")}
    process_response = client.post(
        f"/api/pipelines/{pipeline_id}/process",
        files=files
    )
    task_id = process_response.json()["task_id"]
    
    # Get task status
    status_response = client.get(f"/api/tasks/{task_id}")
    assert status_response.status_code == 200
    data = status_response.json()
    
    assert "task_id" in data
    assert "status" in data
    assert data["task_id"] == task_id


def test_get_nonexistent_task():
    """Test getting non-existent task returns 404"""
    response = client.get("/api/tasks/nonexistent-task-id")
    assert response.status_code == 404


# ==========================================
# STATISTICS TESTS
# ==========================================

def test_get_statistics():
    """Test getting processing statistics"""
    response = client.get("/api/statistics")
    assert response.status_code == 200
    data = response.json()
    
    assert "total_files" in data
    assert "total_chunks" in data
    assert isinstance(data["total_files"], int)


def test_list_processed_files():
    """Test listing processed files"""
    response = client.get("/api/files")
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)


# ==========================================
# INTEGRATION TESTS
# ==========================================

def test_full_pipeline_workflow(sample_pipeline_config, sample_pdf_file):
    """Test complete workflow from pipeline creation to file processing"""
    # Step 1: Create pipeline
    create_response = client.post("/api/pipelines", json=sample_pipeline_config)
    assert create_response.status_code == 200
    pipeline_id = create_response.json()["pipeline_id"]
    
    # Step 2: Validate pipeline
    validate_response = client.post(f"/api/pipelines/{pipeline_id}/validate")
    assert validate_response.status_code == 200
    
    # Step 3: Process file
    files = {"file": ("test.pdf", sample_pdf_file, "application/pdf")}
    process_response = client.post(
        f"/api/pipelines/{pipeline_id}/process",
        files=files
    )
    assert process_response.status_code == 200
    task_id = process_response.json()["task_id"]
    
    # Step 4: Check task status
    status_response = client.get(f"/api/tasks/{task_id}")
    assert status_response.status_code == 200
    
    # Step 5: Verify statistics updated
    stats_response = client.get("/api/statistics")
    assert stats_response.status_code == 200


# ==========================================
# PERFORMANCE TESTS
# ==========================================

@pytest.mark.slow
def test_concurrent_file_processing(sample_pipeline_config):
    """Test processing multiple files concurrently"""
    import concurrent.futures
    
    # Create pipeline
    create_response = client.post("/api/pipelines", json=sample_pipeline_config)
    pipeline_id = create_response.json()["pipeline_id"]
    
    def process_file(file_index):
        file_content = io.BytesIO(f"Mock PDF content {file_index}".encode())
        files = {"file": (f"test_{file_index}.pdf", file_content, "application/pdf")}
        response = client.post(
            f"/api/pipelines/{pipeline_id}/process",
            files=files
        )
        return response.status_code
    
    # Process 10 files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_file, range(10)))
    
    # All should succeed or fail gracefully
    assert all(status in [200, 409] for status in results)


# ==========================================
# ERROR HANDLING TESTS
# ==========================================

def test_invalid_json():
    """Test handling of invalid JSON"""
    response = client.post(
        "/api/pipelines",
        data="invalid json",
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_missing_required_fields():
    """Test handling of missing required fields"""
    incomplete_config = {
        "name": "test"
        # Missing loader, splitter, embedding, database
    }
    
    response = client.post("/api/pipelines", json=incomplete_config)
    assert response.status_code == 422


def test_invalid_parameter_types():
    """Test handling of invalid parameter types"""
    invalid_config = {
        "name": "test",
        "loader": {"type": "pdf"},
        "splitter": {
            "type": "recursive",
            "chunk_size": "not_a_number",  # Should be int
        },
        "embedding": {"type": "huggingface"},
        "database": {"type": "chroma"}
    }
    
    response = client.post("/api/pipelines", json=invalid_config)
    assert response.status_code == 422


# ==========================================
# CONFTEST.PY EXAMPLE
# ==========================================

"""
# tests/conftest.py

import pytest
from app.main import app
from app.services.database import DatabaseManager

@pytest.fixture(scope="session")
def test_db():
    # Create test database
    db = DatabaseManager("test_rag_data.db")
    yield db
    # Cleanup
    import os
    os.remove("test_rag_data.db")

@pytest.fixture(autouse=True)
def reset_db(test_db):
    # Reset database state before each test
    with test_db.get_connection() as conn:
        conn.execute("DELETE FROM files")
        conn.execute("DELETE FROM pipelines")
        conn.execute("DELETE FROM tasks")
"""