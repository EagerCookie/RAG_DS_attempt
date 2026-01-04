# RAGAS - Быстрая шпаргалка

## 🚀 Быстрый старт

### 1. Установка
```bash
pip install ragas datasets
```

### 2. Минимальный пример
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# Ваши данные
data = [{
    "question": "Вопрос?",
    "answer": "Ответ системы"/,
    "contexts": ["Контекст 1", "Контекст 2"]
}]

# Оценка
result = evaluate(
    Dataset.from_list(data),
    metrics=[faithfulness, answer_relevancy]
)
print(result)
```

---

## 📊 Метрики

### БЕЗ ground truth
| Метрика | Что измеряет | Норма |
|---------|--------------|-------|
| **faithfulness** | Нет галлюцинаций | > 0.85 |
| **answer_relevancy** | Ответ по теме | > 0.80 |

### С ground truth
| Метрика | Что измеряет | Норма |
|---------|--------------|-------|
| **context_precision** | Нет шума в контексте | > 0.75 |
| **context_recall** | Весь контекст найден | > 0.80 |
| **answer_similarity** | Похож на эталон | > 0.85 |
| **answer_correctness** | Правильный ответ | > 0.80 |

---

## 🔧 Сравнение конфигураций

### Сравнить разные k
```python
for k in [3, 5, 10]:
    docs = vector_store.similarity_search(query, k=k)
    # ... evaluate ...
```

### Сравнить разные embeddings
```python
models = ["model1", "model2", "model3"]
for model in models:
    embeddings = HuggingFaceEmbeddings(model_name=model)
    # ... evaluate ...
```

### Сравнить разные chunking
```python
sizes = [256, 512, 1024]
for size in sizes:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size)
    # ... evaluate ...
```

---

## 🎯 Интерпретация

### Отлично (Production-ready)
```
faithfulness:       > 0.85
answer_relevancy:   > 0.80
context_precision:  > 0.75
context_recall:     > 0.80
```

### Плохо (Требует доработки)
```
Любая метрика < 0.60
```

---

## 🛠️ Решение проблем

| Проблема | Решение |
|----------|---------|
| **Низкий faithfulness** | Строже промпт, temperature=0 |
| **Низкий context_precision** | Уменьшить k, использовать re-ranking |
| **Низкий context_recall** | Увеличить k, улучшить chunking |
| **Низкий answer_relevancy** | Улучшить промпт, лучшая модель |

---

## 📝 Формат данных

### Минимальный (без ground truth)
```python
{
    "question": "...",
    "answer": "...",
    "contexts": ["...", "..."]  # Список строк!
}
```

### Полный (с ground truth)
```python
{
    "question": "...",
    "answer": "...",
    "contexts": ["...", "..."],
    "ground_truth": "..."  # Эталонный ответ
}
```

---

## 💡 Best Practices

1. ✅ Начните с 20-30 тестовых вопросов
2. ✅ Используйте temperature=0 для оценки
3. ✅ Создавайте ground truth с экспертами
4. ✅ Оценивайте после каждого изменения
5. ✅ Мониторьте качество в продакшене

---

## 🔗 Полная документация

📖 [RAGAS_GUIDE.md](./RAGAS_GUIDE.md) - Подробное руководство  
💻 [ragas_eval.py](./ragas_eval.py) - Примеры кода  
🌐 [docs.ragas.io](https://docs.ragas.io/) - Официальная документация
