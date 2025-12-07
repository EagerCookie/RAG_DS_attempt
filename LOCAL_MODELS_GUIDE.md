# Как использовать локальные модели из папки transformers_models

## 📁 Структура папки transformers_models

Модели HuggingFace хранятся в формате:
```
transformers_models/
├── models--{author}--{model-name}/
│   └── snapshots/
│       └── {hash}/
│           ├── config.json       ← Здесь название модели
│           ├── model.safetensors
│           ├── tokenizer.json
│           └── ...
```

## 🔍 Как узнать название модели

### Способ 1: Посмотреть в config.json

1. **Найдите папку модели:**
   ```
   transformers_models/models--cointegrated--rubert-tiny2/
   ```

2. **Откройте `config.json`:**
   ```bash
   # Windows
   type "transformers_models\models--cointegrated--rubert-tiny2\snapshots\{hash}\config.json"
   
   # Linux/Mac
   cat transformers_models/models--cointegrated--rubert-tiny2/snapshots/{hash}/config.json
   ```

3. **Найдите поле `_name_or_path`:**
   ```json
   {
     "_name_or_path": "cointegrated/rubert-tiny2",  ← Это название!
     ...
   }
   ```

### Способ 2: По имени папки

Имя папки содержит название модели:
```
models--{author}--{model-name}
         ↓         ↓
models--cointegrated--rubert-tiny2
```

**Название модели:** `cointegrated/rubert-tiny2`

## 📋 Ваши доступные модели

Из вашей папки `transformers_models`:

### 1. **rubert-tiny2**
- **Папка:** `models--cointegrated--rubert-tiny2`
- **Название для API:** `cointegrated/rubert-tiny2`
- **Описание:** Компактная русская BERT модель для эмбеддингов
- **Размер:** ~112 MB
- **Использование:** Embedding модель для русского языка

### 2. **USER-bge-m3**
- **Папка:** `models--DeepVk--USER-bge-m3`
- **Название для API:** `DeepVk/USER-bge-m3`
- **Описание:** Мультиязычная модель эмбеддингов
- **Использование:** Embedding модель (поддержка русского и других языков)

## 🚀 Как использовать в пайплайне

### В интерфейсе create_pipeline.html:

1. **Выберите тип эмбеддинга:** HuggingFace
2. **В поле "Model Name" укажите:**
   ```
   cointegrated/rubert-tiny2
   ```
   или
   ```
   DeepVk/USER-bge-m3
   ```

3. **Укажите путь к кэшу (опционально):**
   ```
   ./transformers_models
   ```

### Пример конфигурации:

```json
{
  "embedding": {
    "type": "huggingface",
    "model_name": "cointegrated/rubert-tiny2",
    "device": "cpu",
    "normalize_embeddings": true,
    "cache_folder": "./transformers_models"
  }
}
```

## 💡 Важные моменты

### 1. **Cache Folder**
Если вы указываете `cache_folder`, HuggingFace будет искать модель там:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="cointegrated/rubert-tiny2",
    cache_folder="./transformers_models"  # Ваша папка
)
```

### 2. **Автоматическое определение**
HuggingFace автоматически найдет модель в кэше по названию:
- `cointegrated/rubert-tiny2` → `models--cointegrated--rubert-tiny2`
- `DeepVk/USER-bge-m3` → `models--DeepVk--USER-bge-m3`

### 3. **Проверка модели**
Чтобы проверить что модель работает:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="cointegrated/rubert-tiny2",
    cache_folder="./transformers_models"
)

# Тест
result = embeddings.embed_query("Привет, мир!")
print(f"Размерность вектора: {len(result)}")
```

## 🎯 Быстрая справка

| Папка | Название для API | Язык | Размер вектора |
|-------|------------------|------|----------------|
| `models--cointegrated--rubert-tiny2` | `cointegrated/rubert-tiny2` | RU | 312 |
| `models--DeepVk--USER-bge-m3` | `DeepVk/USER-bge-m3` | Multi | 1024 |

## ✅ Готово!

Теперь вы знаете как использовать ваши локальные модели в пайплайне! 🚀

### Пример создания пайплайна с rubert-tiny2:

1. Откройте `create_pipeline.html`
2. Выберите **HuggingFace** в разделе эмбеддингов
3. Укажите **Model Name:** `cointegrated/rubert-tiny2`
4. Укажите **Cache Folder:** `./transformers_models`
5. Создайте пайплайн!
