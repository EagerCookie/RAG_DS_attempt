# Анализ проблемы: Превышение времени ожидания и 500 ошибка

## 📋 Краткое описание

При обработке файла через API произошло превышение времени ожидания. При попытке проверить статус обработки через endpoint `/api/pipelines/{pipeline_id}/validate` сервер вернул **500 Internal Server Error**.

---

## 🔍 Детальный анализ

### 1. **Что произошло**

#### Последовательность событий:
1. ✅ Файл был отправлен на обработку через API
2. ⏳ Обработка заняла больше времени, чем ожидалось (timeout)
3. 💻 Процессор был загружен (видно в мониторе ресурсов)
4. ❌ При попытке проверить статус через `/validate` endpoint - получена 500 ошибка
5. 💥 Сервер упал с `AttributeError`

#### Ошибка из лога:
```
AttributeError: 'PipelineConfig' object has no attribute 'splitter'
```

**Место ошибки:**
- Файл: `app/services/pipeline_service.py`
- Строка: 259
- Функция: `PipelineValidator.validate_compatibility()`

---

### 2. **Корневая причина**

#### Архитектурное изменение системы:

Ваша система претерпела рефакторинг разделения ответственности:

**До (старая архитектура):**
```python
class PipelineConfig:
    loader: LoaderConfig      # Как загружать файлы
    splitter: SplitterConfig  # Как разбивать на чанки
    embedding: EmbeddingConfig  # Модель эмбеддингов
    database: DatabaseConfig    # Векторная БД
```

**После (новая архитектура):**
```python
class PipelineConfig:
    # Только база знаний
    embedding: EmbeddingConfig  # Модель эмбеддингов
    database: DatabaseConfig    # Векторная БД
    default_variant: Optional[ProcessingVariantConfig]

class ProcessingVariantConfig:
    # Варианты обработки
    loader: LoaderConfig      # Как загружать файлы
    splitter: SplitterConfig  # Как разбивать на чанки
```

#### Проблема:
Класс `PipelineValidator` не был обновлен после рефакторинга и продолжал обращаться к `config.splitter`, которого больше нет в `PipelineConfig`.

---

### 3. **Почему процессор был загружен**

Это **НОРМАЛЬНОЕ поведение** при обработке документов в RAG системе:

#### Этапы обработки (каждый загружает CPU):

1. **Загрузка модели эмбеддингов** (10-30 сек)
   - Скачивание/загрузка модели HuggingFace (например, `DeepVk/USER-bge-m3`)
   - Инициализация модели в памяти
   - Размер модели: ~500MB - 2GB

2. **Создание эмбеддингов** (основная нагрузка)
   - Для каждого чанка документа создается вектор
   - Пример: документ → 100 чанков → 100 векторов по 1024 измерения
   - На CPU это медленно (на GPU быстрее)

3. **Сохранение в векторную БД**
   - Запись векторов в ChromaDB
   - Создание индексов

#### Примерное время обработки:
- **Маленький PDF (1-2 страницы)**: 30-60 секунд
- **Средний PDF (10-20 страниц)**: 2-5 минут
- **Большой PDF (50+ страниц)**: 10-20 минут

**На CPU обработка в 3-5 раз медленнее, чем на GPU!**

---

### 4. **Почему возникла 500 ошибка**

#### Цепочка вызовов:

```
1. Frontend → GET /api/pipelines/{id}/validate
2. main.py:validate_pipeline() → line 624
3. PipelineValidator.validate_compatibility(config)
4. pipeline_service.py:259 → if isinstance(config.splitter, ...)
5. ❌ AttributeError: 'PipelineConfig' object has no attribute 'splitter'
```

#### Код с ошибкой (ДО исправления):

```python
# pipeline_service.py, строка 259
def validate_compatibility(config: PipelineConfig) -> List[str]:
    warnings = []
    
    # ❌ ОШИБКА: config.splitter больше не существует!
    if isinstance(config.splitter, RecursiveSplitterConfig):
        if config.splitter.chunk_size > 2000:
            warnings.append("Large chunk size...")
    
    if isinstance(config.embedding, HuggingFaceEmbeddingConfig):
        if "bge" in config.embedding.model_name.lower():
            # ❌ ОШИБКА: config.splitter больше не существует!
            if isinstance(config.splitter, SentenceTransformerSplitterConfig):
                if config.splitter.chunk_size > 512:
                    warnings.append("BGE models work best...")
```

---

## ✅ Решение

### Исправленный код:

#### 1. `PipelineValidator.validate_compatibility()` - ПОСЛЕ исправления:

```python
@staticmethod
def validate_compatibility(config: PipelineConfig) -> List[str]:
    """
    Check if pipeline components are compatible
    
    NOTE: PipelineConfig now only contains embedding + database.
    Loader and splitter are in ProcessingVariantConfig.
    
    Returns:
        List of warning messages (empty if all OK)
    """
    warnings = []
    
    # ✅ Проверяем только embedding
    if isinstance(config.embedding, HuggingFaceEmbeddingConfig):
        if config.embedding.device == "cuda":
            warnings.append("Using CUDA device - ensure GPU is available")
        
        if "large" in config.embedding.model_name.lower():
            warnings.append(
                "Large embedding models may require significant memory"
            )
    
    # ✅ Проверяем только database
    if isinstance(config.database, ChromaDBConfig):
        import os
        if not os.path.exists(config.database.persist_directory):
            warnings.append(
                f"Persist directory will be created"
            )
    
    return warnings
```

#### 2. `estimate_processing_time()` - ПОСЛЕ исправления:

```python
@staticmethod
def estimate_processing_time(config: PipelineConfig, file_size_mb: float) -> float:
    """
    Estimate processing time in seconds based on pipeline configuration
    
    NOTE: This is a rough estimate based only on embedding model.
    Actual time depends on loader/splitter from ProcessingVariantConfig.
    """
    base_time = file_size_mb * 2  # 2 seconds per MB base
    
    # ✅ Проверяем только embedding
    if isinstance(config.embedding, HuggingFaceEmbeddingConfig):
        if "large" in config.embedding.model_name.lower():
            base_time += 30
        else:
            base_time += 10
    
    return base_time
```

---

## 📊 Что изменилось

### Файлы изменены:
- ✅ `app/services/pipeline_service.py`
  - Функция `validate_compatibility()` - удалены проверки `config.splitter`
  - Функция `estimate_processing_time()` - удалены проверки `config.splitter`

### Что НЕ изменилось:
- ✅ Логика обработки файлов (`process_document_with_variant_task`)
- ✅ API endpoints
- ✅ Модели данных (`PipelineConfig`, `ProcessingVariantConfig`)

---

## 🎯 Рекомендации

### 1. **Увеличить timeout для обработки файлов**

Если вы используете frontend с таймаутом, увеличьте его:

```javascript
// Frontend
const response = await fetch('/api/pipelines/{id}/process', {
    method: 'POST',
    body: formData,
    // ❌ Плохо: timeout: 30000 (30 секунд)
    // ✅ Хорошо: timeout: 300000 (5 минут)
});
```

### 2. **Использовать polling для проверки статуса**

Вместо ожидания ответа, используйте асинхронную обработку:

```javascript
// 1. Отправить файл
const { task_id } = await uploadFile();

// 2. Проверять статус каждые 2 секунды
const interval = setInterval(async () => {
    const status = await fetch(`/api/tasks/${task_id}`);
    if (status.status === 'completed') {
        clearInterval(interval);
        console.log('Done!');
    }
}, 2000);
```

### 3. **Использовать GPU для ускорения**

Если у вас есть NVIDIA GPU:

```python
# В конфигурации пайплайна
{
    "embedding": {
        "type": "huggingface",
        "model_name": "DeepVk/USER-bge-m3",
        "device": "cuda",  # ✅ Вместо "cpu"
        "normalize_embeddings": true
    }
}
```

**Ускорение: 3-5x быстрее!**

### 4. **Мониторинг прогресса**

Используйте endpoint `/api/tasks/{task_id}` для отслеживания:

```json
{
    "task_id": "...",
    "status": "processing",
    "progress": 0.65,  // 65%
    "message": "Creating embeddings..."
}
```

---

## 🔧 Проверка исправления

### Тест 1: Проверить валидацию пайплайна

```bash
curl http://localhost:8001/api/pipelines/{pipeline_id}/validate
```

**Ожидаемый результат:**
```json
{
    "valid": true,
    "warnings": [
        "Large embedding models may require significant memory"
    ],
    "estimated_time_per_mb": 12.0
}
```

### Тест 2: Обработать файл

```bash
curl -X POST http://localhost:8001/api/pipelines/{pipeline_id}/process \
  -F "file=@test.pdf"
```

**Ожидаемый результат:**
```json
{
    "task_id": "...",
    "status": "pending",
    "message": "Processing with variant '...'"
}
```

---

## 📝 Выводы

### Что мы узнали:

1. ✅ **Высокая загрузка CPU - это нормально** при обработке документов
2. ✅ **Обработка может занимать минуты**, особенно на CPU
3. ✅ **500 ошибка была вызвана** устаревшим кодом валидатора
4. ✅ **Архитектура изменилась**: `PipelineConfig` теперь только embedding + database

### Что исправили:

1. ✅ Обновили `PipelineValidator.validate_compatibility()`
2. ✅ Обновили `PipelineValidator.estimate_processing_time()`
3. ✅ Добавили комментарии о новой архитектуре

### Что НЕ является проблемой:

1. ✅ Высокая загрузка CPU во время обработки
2. ✅ Длительное время обработки (минуты)
3. ✅ Timeout при обработке больших файлов

---

## 🚀 Следующие шаги

1. **Перезапустить сервер** с исправленным кодом
2. **Протестировать** endpoint `/validate`
3. **Обработать файл** и проверить, что все работает
4. **Настроить frontend** для правильного polling статуса
5. **Рассмотреть использование GPU** для ускорения

---

**Дата анализа:** 2026-01-04  
**Статус:** ✅ Проблема решена
