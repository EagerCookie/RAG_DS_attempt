# RAG System - TODO & Roadmap

## ✅ Выполнено

### Архитектура
- ✅ **Рефакторинг архитектуры** - Разделение Pipeline (база знаний) и Variant (способ обработки)
- ✅ **Переиспользование пайплайнов** - Возможность использовать разные loader/splitter с одной векторной БД
- ✅ **Tracking variant_id** - Отслеживание какой вариант использовался для обработки файла
- ✅ **Unified processing endpoint** - Единый endpoint `/api/pipelines/{id}/process` с опциональным `variant_id`

### API Endpoints
- ✅ **Variant Management API**:
  - `POST /api/pipelines/{id}/variants` - Создание нового варианта
  - `GET /api/pipelines/{id}/variants` - Список вариантов
  - `GET /api/variants/{id}` - Детали варианта
  - `DELETE /api/variants/{id}` - Удаление варианта
- ✅ **RAG Inference API** - `/api/rag/query` с поддержкой всех LLM провайдеров
- ✅ **LLM Integration** - Поддержка OpenAI, Anthropic, DeepSeek, Custom API (LM Studio, Ollama)

### UI/UX
- ✅ **create_pipeline.html** - Обновлен под новую архитектуру с режимами "Создать новый" и "Использовать существующий"
- ✅ **rag_inference.html** - Интерфейс для RAG запросов с поддержкой:
  - Выбор пайплайна
  - Выбор LLM провайдера и модели
  - Настройка temperature (для LLM) и top_k (для выдачи документов эмбеддера)
  - Custom API URL и model ID для локальных моделей

### Документация
- ✅ **REFACTORING.md** - Описание новой архитектуры
- ✅ **PIPELINE_SELECTOR_GUIDE.md** - Инструкция по работе с режимами
- ✅ **LM_STUDIO_SETUP.md** - Настройка локальных LLM через LM Studio
- ✅ **LOCAL_MODELS_GUIDE.md** - Работа с локальными embedding моделями

---

## 🔴 Критические приоритеты

### 1. Очередь задач и параллелизм
**Проблема:**
- Обработка происходит последовательно в background tasks
- Нет контроля параллелизма
- При перезапуске сервера задачи теряются

**Решение:**
```python
# Использовать Celery + Redis (уже есть в docker-compose.yml)
# 1. Установить зависимости
pip install celery redis

# 2. Создать app/celery_app.py
from celery import Celery

celery_app = Celery(
    'rag_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# 3. Переписать process_document_with_variant_task как Celery task
@celery_app.task(bind=True)
def process_document_task(self, task_id, pipeline_id, variant_id, ...):
    # Existing logic
    self.update_state(state='PROGRESS', meta={'progress': 0.5})
```

**Преимущества:**
- ✅ Персистентность задач
- ✅ Контроль параллелизма (workers)
- ✅ Retry механизм
- ✅ Task monitoring

### 2. Аутентификация и авторизация
**Проблема:**
- API открыт для всех
- Нет разграничения доступа

**Решение:**
```python
# 1. JWT Authentication
from fastapi_jwt_auth import AuthJWT

@app.post("/api/auth/login")
async def login(credentials: LoginRequest):
    # Verify credentials
    access_token = Authorize.create_access_token(subject=user.id)
    return {"access_token": access_token}

# 2. Protect endpoints
@app.post("/api/pipelines")
async def create_pipeline(
    config: PipelineConfig,
    Authorize: AuthJWT = Depends()
):
    Authorize.jwt_required()
    current_user = Authorize.get_jwt_subject()
    # ...
```

**Альтернатива - API Keys:**
```python
# Простой вариант для начала
API_KEYS = set(os.getenv("API_KEYS", "").split(","))

async def verify_api_key(api_key: str = Header(...)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401)
```

### 3. Global Exception Handling
**Проблема:**
- Некоторые исключения могут не обрабатываться
- Нет единого формата ошибок

**Решение:**
```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if DEBUG else "An error occurred",
            "request_id": request.state.request_id
        }
    )

# Middleware для request_id
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response
```

---

## 🟡 Средние приоритеты

### 4. Batch Upload
**Текущее состояние:** Загрузка только одного файла

**Решение:**
```python
@app.post("/api/pipelines/{pipeline_id}/process/batch")
async def process_files_batch(
    pipeline_id: str,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    variant_id: Optional[str] = None
):
    task_ids = []
    for file in files:
        task_id = str(uuid.uuid4())
        # Create task for each file
        background_tasks.add_task(
            process_document_with_variant_task,
            task_id, pipeline_id, variant_id, ...
        )
        task_ids.append(task_id)
    
    return {"batch_id": str(uuid.uuid4()), "task_ids": task_ids}
```

### 5. RAG Metrics (RAGAS Integration)
**Файл уже есть:** `ragas_eval.py`

**TODO:**
- [ ] Создать endpoint `/api/rag/evaluate`
- [ ] Интегрировать метрики:
  - `faithfulness` - нет ли в ответе утверждений без контекста
  - `answer_relevancy` - насколько ответ релевантен вопросу
  - `context_precision` - качество найденного контекста
  - `context_recall` - полнота найденного контекста
- [ ] Добавить UI для просмотра метрик
- [ ] Сохранять метрики в БД для анализа

**Пример:**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

@app.post("/api/rag/evaluate")
async def evaluate_rag(request: EvaluateRequest):
    # Run evaluation
    result = evaluate(
        dataset=request.dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )
    return result
```

### 6. Расширение типов документов
**Текущие:** PDF, TXT

**Добавить:**
```python
# app/services/loaders.py

class DOCXLoaderConfig(LoaderConfig):
    type: Literal["docx"] = "docx"

class MarkdownLoaderConfig(LoaderConfig):
    type: Literal["markdown"] = "markdown"

class HTMLLoaderConfig(LoaderConfig):
    type: Literal["html"] = "html"
    
class CSVLoaderConfig(LoaderConfig):
    type: Literal["csv"] = "csv"
    encoding: str = "utf-8"
    delimiter: str = ","

# В LoaderFactory
elif config.type == "docx":
    from langchain_community.document_loaders import Docx2txtLoader
    return Docx2txtLoader(file_path)
```

### 7. Rate Limiting
**Проблема:** Возможна перегрузка API

**Решение:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/rag/query")
@limiter.limit("10/minute")  # 10 запросов в минуту
async def rag_query(request: Request, query: RAGQueryRequest):
    # ...
```

---

## 🟢 Минорные улучшения

### 8. Structured Logging
**Решение:**
```python
import structlog

logger = structlog.get_logger()

logger.info(
    "pipeline_created",
    pipeline_id=pipeline_id,
    user_id=user_id,
    config=config.dict()
)

# Интеграция с ELK stack для анализа
```

### 9. Мониторинг и метрики
**Решение:**
```python
from prometheus_fastapi_instrumentator import Instrumentator

# Автоматические метрики
Instrumentator().instrument(app).expose(app)

# Custom метрики
from prometheus_client import Counter, Histogram

documents_processed = Counter(
    'documents_processed_total',
    'Total documents processed',
    ['pipeline_id', 'variant_id', 'status']
)

processing_time = Histogram(
    'document_processing_seconds',
    'Time spent processing documents'
)
```

**Визуализация:** Grafana dashboards

### 10. Environment Variables
**Текущее состояние:** Некоторые параметры хардкодятся

**Решение:**
```python
# .env
DATABASE_PATH=./data/rag_system.db
CHROMA_PERSIST_DIR=./chroma_db
EMBEDDING_CACHE_DIR=./transformers_models
MAX_WORKERS=4
CELERY_BROKER_URL=redis://localhost:6379/0
LOG_LEVEL=INFO

# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_path: str
    chroma_persist_dir: str
    embedding_cache_dir: str
    max_workers: int = 4
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 🚀 Будущие возможности

### 11. Telegram Bot Integration
**Цель:** Интерфейс для RAG через Telegram

**Решение:**
```python
# bot/telegram_bot.py
from aiogram import Bot, Dispatcher, types

@dp.message_handler(commands=['ask'])
async def ask_question(message: types.Message):
    # Call RAG API
    response = await call_rag_api(
        pipeline_id=user_pipeline,
        query=message.text
    )
    await message.answer(response['answer'])
```

### 12. Интерактивная документация
**Цель:** Swagger UI с примерами

**Уже есть:** FastAPI автоматически генерирует `/docs`

**Улучшения:**
- Добавить примеры в docstrings
- Настроить OpenAPI schema
- Добавить Redoc (`/redoc`)

### 13. Multi-collection support
**Текущее:** Каждый pipeline = своя БД

**Рассмотреть:** Использование коллекций в одной БД
```python
# Вместо разных persist_directory
# Использовать разные collection_name в одной ChromaDB
```

### 14. Референсные файлы для метрик
**Цель:** Тестовые датасеты для оценки качества

**Создать:**
- `test_datasets/` - папка с тестовыми вопросами и ответами
- Ground truth для метрик
- Benchmark pipelines

---

## 📊 Приоритизация

### Фаза 1 (Критическая стабильность)
1. ✅ Рефакторинг архитектуры
2. 🔴 Celery + Redis для очереди задач
3. 🔴 Global exception handling
4. 🔴 Базовая аутентификация (API keys)

### Фаза 2 (Функциональность)
5. 🟡 Batch upload
6. 🟡 RAG metrics integration
7. 🟡 Расширение типов документов
8. 🟡 Rate limiting

### Фаза 3 (Операционная готовность)
9. 🟢 Structured logging
10. 🟢 Prometheus + Grafana
11. 🟢 Environment variables
12. 🟢 Полная документация

### Фаза 4 (Расширения)
13. 🚀 Telegram bot
14. 🚀 Multi-collection support
15. 🚀 Advanced RAG techniques (re-ranking, hybrid search)

---

## 📝 Заметки

- **Docker готов:** `docker-compose.yml` уже содержит Redis и Chroma
- **RAGAS готов:** `ragas_eval.py` можно интегрировать
- **Архитектура гибкая:** Легко добавлять новые loaders/splitters/embeddings
- **API документирован:** FastAPI Swagger UI доступен на `/docs`

---

**Последнее обновление:** 2025-12-07
**Версия системы:** 2.0 (после рефакторинга)