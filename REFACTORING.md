# Рефакторинг RAG System - Новая архитектура

## 🎯 Основные изменения

### 1. **Новая концепция Pipeline и Variant**

**До:**
- Pipeline содержал: loader + splitter + embedding + database
- Создавались дублирующиеся пайплайны для разных форматов файлов

**После:**
- **Pipeline** = только **embedding + database** (база знаний)
- **Variant** = **loader + splitter** (способ обработки документов)
- Один pipeline может иметь множество вариантов обработки

### 2. **Упрощенный API**

**До:**
- `/api/pipelines/{id}/process` - обработка с настройками пайплайна
- `/api/variants/{id}/process` - обработка с вариантом

**После:**
- `/api/pipelines/{id}/process?variant_id=xxx` - единый endpoint
- Если `variant_id` не указан, используется первый доступный вариант

### 3. **Отслеживание variant_id**

Теперь во всех файлах сохраняется `variant_id`, что позволяет:
- Видеть каким вариантом был обработан файл
- Фильтровать файлы по варианту
- Анализировать эффективность разных вариантов обработки

---

## 📋 Новые модели данных

### PipelineConfig (configs.py)
```python
class PipelineConfig(BaseModel):
    name: str
    embedding: EmbeddingConfig  # Только embedding
    database: DatabaseConfig    # Только database
    default_variant: Optional[ProcessingVariantConfig]  # Опциональный дефолтный вариант
```

### ProcessingVariantConfig (configs.py)
```python
class ProcessingVariantConfig(BaseModel):
    name: str
    loader: LoaderConfig     # Loader для варианта
    splitter: SplitterConfig # Splitter для варианта
    description: Optional[str]
```

### PipelineResponse (schemas.py)
```python
class PipelineResponse(BaseModel):
    pipeline_id: str
    config: PipelineConfig
    created_at: datetime
    variants: Optional[List[ProcessingVariantResponse]]  # Список вариантов
    default_variant_id: Optional[str]
```

### ProcessingStatus (schemas.py)
```python
class ProcessingStatus(BaseModel):
    task_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    progress: Optional[float]
    message: Optional[str]
    error: Optional[str]
    variant_id: Optional[str]  # Какой вариант использовался
```

---

## 🔄 Обновленные API Endpoints

### Pipelines

**POST /api/pipelines**
```json
{
  "name": "My Knowledge Base",
  "embedding": {
    "type": "huggingface",
    "model_name": "DeepVk/USER-bge-m3"
  },
  "database": {
    "type": "chroma",
    "collection_name": "my_docs"
  },
  "default_variant": {  // Опционально
    "name": "PDF Processor",
    "loader": {"type": "pdf"},
    "splitter": {"type": "recursive", "chunk_size": 1000}
  }
}
```

**GET /api/pipelines/{pipeline_id}**
- Возвращает pipeline + список всех вариантов

### Variants

**POST /api/pipelines/{pipeline_id}/variants**
```json
{
  "name": "Text File Processor",
  "loader": {"type": "text", "encoding": "utf-8"},
  "splitter": {"type": "recursive", "chunk_size": 500},
  "description": "For processing plain text files"
}
```

**GET /api/pipelines/{pipeline_id}/variants**
- Список всех вариантов пайплайна

**GET /api/variants/{variant_id}**
- Детали конкретного варианта + количество обработанных файлов

**DELETE /api/variants/{variant_id}**
- Удалить вариант (файлы остаются)

### Processing

**POST /api/pipelines/{pipeline_id}/process?variant_id={variant_id}**
- `variant_id` опционален
- Если не указан, используется первый доступный вариант
- Возвращает `task_id`, `variant_id`, `variant_name`

### Files

**GET /api/pipelines/{pipeline_id}/files**
- Теперь включает `variant_id` для каждого файла

**GET /api/vector-databases/{vector_db_identifier}/files**
- Теперь включает `variant_id` для каждого файла

**GET /api/files**
- Все файлы теперь с `variant_id`

---

## 🗄️ Изменения в базе данных

### Таблица `files`
Добавлена колонка:
```sql
variant_id TEXT  -- ID варианта, которым был обработан файл
```

### Таблица `processing_variants`
```sql
CREATE TABLE processing_variants (
    id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    name TEXT NOT NULL,
    config TEXT NOT NULL,  -- JSON ProcessingVariantConfig
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pipeline_id) REFERENCES pipelines(id) ON DELETE CASCADE
)
```

---

## 💡 Примеры использования

### Создание пайплайна с дефолтным вариантом
```python
pipeline = {
    "name": "Research Papers DB",
    "embedding": {
        "type": "huggingface",
        "model_name": "DeepVk/USER-bge-m3"
    },
    "database": {
        "type": "chroma",
        "collection_name": "research_papers"
    },
    "default_variant": {
        "name": "PDF Research Papers",
        "loader": {"type": "pdf", "extract_images": True},
        "splitter": {"type": "recursive", "chunk_size": 1500, "chunk_overlap": 200}
    }
}
```

### Добавление варианта для другого формата
```python
variant = {
    "name": "Text Research Papers",
    "loader": {"type": "text", "encoding": "utf-8"},
    "splitter": {"type": "recursive", "chunk_size": 1500, "chunk_overlap": 200},
    "description": "For plain text versions of papers"
}

POST /api/pipelines/{pipeline_id}/variants
```

### Обработка файла с конкретным вариантом
```bash
# С указанием варианта
POST /api/pipelines/{pipeline_id}/process?variant_id={variant_id}

# Без указания (используется первый доступный)
POST /api/pipelines/{pipeline_id}/process
```

---

## ✅ Преимущества новой архитектуры

1. **Переиспользование базы знаний** - один pipeline для разных форматов файлов
2. **Единый API** - один endpoint для обработки вместо двух
3. **Прозрачность** - всегда видно каким вариантом обработан файл
4. **Гибкость** - легко добавлять новые варианты обработки
5. **Логичность** - четкое разделение: база знаний (pipeline) vs способ обработки (variant)

---

## 🔧 Обратная совместимость

**BREAKING CHANGES:**
- `PipelineConfig` больше не содержит `loader` и `splitter` напрямую
- Старые пайплайны нужно мигрировать: создать variant из loader+splitter

**Миграция:**
1. Для каждого старого пайплайна создать variant с его loader+splitter
2. Обновить PipelineConfig, убрав loader и splitter
3. Обновить код обработки для использования вариантов
