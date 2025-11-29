# RAG Pipeline API - Project Structure

## 📁 Структура проекта

```
rag-pipeline-api/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py        # Dependency injection
│   │   └── endpoints/
│   │       ├── __init__.py
│   │       ├── loaders.py
│   │       ├── splitters.py
│   │       ├── embeddings.py
│   │       ├── databases.py
│   │       ├── pipelines.py
│   │       └── files.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # App settings
│   │   └── registry.py            # Component registry
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── configs.py             # Pydantic config models
│   │   └── schemas.py             # API response schemas
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── database.py            # DatabaseManager
│   │   ├── pipeline_service.py   # Pipeline processing
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── pdf_loader.py
│   │   │   └── text_loader.py
│   │   ├── splitters/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── implementations.py
│   │   ├── embeddings/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── huggingface.py
│   │   └── databases/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── chroma.py
│   │       └── qdrant.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── hashing.py
│
├── frontend/                      # React/Vue frontend (optional)
│   ├── package.json
│   └── src/
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_pipeline.py
│   └── test_database.py
│
├── data/
│   ├── chroma_langchain_db/      # Vector DB storage
│   └── transformers_models/      # Model cache
│
├── .env                          # Environment variables
├── .gitignore
├── requirements.txt
├── README.md
└── run.py                        # Entry point
```

## 🚀 Установка и запуск

### 1. Установка зависимостей

```bash
# Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установите зависимости
pip install -r requirements.txt
```

### 2. requirements.txt

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
python-dotenv==1.0.0

# LangChain
langchain==0.1.0
langchain-community==0.0.10
langchain-text-splitters==0.0.1
langchain-huggingface==0.0.1
langchain-chroma==0.1.0
# langchain-qdrant==0.1.0  # Опционально

# ML/AI
sentence-transformers==2.2.2
chromadb==0.4.22
huggingface-hub==0.19.4

# PDF processing
pypdf==3.17.4

# Other
httpx==0.25.2
aiofiles==23.2.1
```

### 3. Настройка .env

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Database
DATABASE_PATH=./data/rag_data.db

# Vector Databases
CHROMA_PERSIST_DIR=./data/chroma_langchain_db
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Models
TRANSFORMERS_CACHE=./data/transformers_models
DEFAULT_EMBEDDING_MODEL=DeepVk/USER-bge-m3
DEFAULT_DEVICE=cpu

# File Upload
MAX_UPLOAD_SIZE=50  # MB
ALLOWED_EXTENSIONS=pdf,txt,md
```

### 4. Запуск сервера

```bash
# Разработка
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Или через run.py
python run.py
```

### 5. Документация API

После запуска доступна по адресам:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔧 Примеры использования API

### Создание пайплайна

```bash
curl -X POST "http://localhost:8000/api/pipelines" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_pipeline",
    "loader": {
      "type": "pdf",
      "extract_images": true
    },
    "splitter": {
      "type": "recursive",
      "chunk_size": 1000,
      "chunk_overlap": 200
    },
    "embedding": {
      "type": "huggingface",
      "model_name": "DeepVk/USER-bge-m3",
      "device": "cpu"
    },
    "database": {
      "type": "chroma",
      "collection_name": "my_docs"
    }
  }'
```

### Обработка файла

```bash
curl -X POST "http://localhost:8000/api/pipelines/{pipeline_id}/process" \
  -F "file=@/path/to/document.pdf"
```

### Проверка статуса

```bash
curl "http://localhost:8000/api/tasks/{task_id}"
```

## 🎨 Frontend интеграция

### Пример React компонента

```jsx
import { useState, useEffect } from 'react';

function PipelineBuilder() {
  const [loaders, setLoaders] = useState([]);
  const [pipeline, setPipeline] = useState({
    name: 'my_pipeline',
    loader: { type: 'pdf' },
    splitter: { type: 'recursive', chunk_size: 1000 },
    embedding: { type: 'huggingface', model_name: 'DeepVk/USER-bge-m3' },
    database: { type: 'chroma', collection_name: 'docs' }
  });

  useEffect(() => {
    // Load available loaders
    fetch('http://localhost:8000/api/loaders')
      .then(r => r.json())
      .then(data => setLoaders(data));
  }, []);

  const createPipeline = async () => {
    const response = await fetch('http://localhost:8000/api/pipelines', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(pipeline)
    });
    const data = await response.json();
    return data.pipeline_id;
  };

  const uploadFile = async (pipelineId, file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(
      `http://localhost:8000/api/pipelines/${pipelineId}/process`,
      { method: 'POST', body: formData }
    );
    return response.json();
  };

  return (
    <div>
      {/* UI для настройки пайплайна */}
    </div>
  );
}
```

## 📊 Мониторинг и логирование

### Добавление логирования

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Метрики с Prometheus (опционально)

```python
from prometheus_fastapi_instrumentator import Instrumentator

@app.on_event("startup")
async def startup():
    Instrumentator().instrument(app).expose(app)
```

## 🔒 Production настройки

### 1. CORS для production

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

### 2. Rate limiting

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/pipelines/{pipeline_id}/process")
@limiter.limit("10/minute")
async def process_file(...):
    ...
```

### 3. Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Проверка JWT токена
    ...
```

## 🧪 Тестирование

```bash
# Установка pytest
pip install pytest pytest-asyncio httpx

# Запуск тестов
pytest tests/
```

## 📝 Дополнительные возможности

### Webhook уведомления

```python
@app.post("/api/webhooks/register")
async def register_webhook(url: str, events: List[str]):
    # Регистрация webhook для событий
    ...
```

### Batch processing

```python
@app.post("/api/pipelines/{pipeline_id}/batch")
async def process_batch(pipeline_id: str, files: List[UploadFile]):
    # Обработка множества файлов
    ...
```

### Pipeline templates

```python
@app.get("/api/templates")
async def list_templates():
    return [
        {
            "id": "general_purpose",
            "name": "General Purpose",
            "description": "Balanced settings for most documents",
            "config": {...}
        }
    ]
```