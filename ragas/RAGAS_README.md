# 📚 RAGAS Evaluation - Документация и примеры

Полный набор материалов для изучения и использования RAGAS (RAG Assessment) фреймворка для оценки качества RAG систем.

---

## 📁 Файлы в этой папке

### 📘 Обучающие материалы

1. **[RAGAS_GUIDE.md](./RAGAS_GUIDE.md)** - Полное руководство
   - Введение в RAGAS
   - Подробное описание всех метрик
   - Практические примеры
   - Сравнение конфигураций
   - Best practices
   - Troubleshooting
   
2. **[RAGAS_CHEATSHEET.md](./RAGAS_CHEATSHEET.md)** - Быстрая шпаргалка
   - Быстрый старт
   - Таблица метрик
   - Решение типичных проблем
   - Формат данных

### 💻 Код

3. **[ragas_eval.py](./ragas_eval.py)** - Рабочие примеры
   - Базовая оценка без ground truth
   - Полная оценка с ground truth
   - Детальный анализ результатов
   - Сравнение конфигураций (k, embeddings, chunking)
   - Мониторинг в продакшене
   - Экспорт результатов

### 📊 Тестовые данные

4. **[test_questions_example.json](./test_questions_example.json)** - Пример тестового набора
   - Формат вопросов с ground truth
   - Разные категории (factual, reasoning)
   - Разные уровни сложности

---

## 🚀 Быстрый старт

### 1. Установка

```bash
pip install ragas datasets langchain langchain-openai langchain-huggingface
```

### 2. Настройка API ключа

```bash
# В .env файле
OPENAI_API_KEY=sk-...
```

### 3. Запуск примера

```bash
# Базовый пример
python ragas_eval.py
```

### 4. Изучение документации

1. Начните с **[RAGAS_CHEATSHEET.md](./RAGAS_CHEATSHEET.md)** для быстрого понимания
2. Прочитайте **[RAGAS_GUIDE.md](./RAGAS_GUIDE.md)** для глубокого изучения
3. Изучите **[ragas_eval.py](./ragas_eval.py)** для практических примеров

---

## 📊 Метрики RAGAS

### Метрики БЕЗ ground truth (эталонных ответов)

| Метрика | Описание | Целевое значение |
|---------|----------|------------------|
| **faithfulness** | Проверка на галлюцинации | > 0.85 |
| **answer_relevancy** | Релевантность ответа вопросу | > 0.80 |

### Метрики С ground truth (требуют эталонные ответы)

| Метрика | Описание | Целевое значение |
|---------|----------|------------------|
| **context_precision** | Точность retrieval (нет шума) | > 0.75 |
| **context_recall** | Полнота retrieval (все найдено) | > 0.80 |
| **answer_similarity** | Схожесть с эталоном | > 0.85 |
| **answer_correctness** | Общая корректность | > 0.80 |

---

## 🎯 Типичные сценарии использования

### Сценарий 1: Первичная оценка системы

```python
# Используйте базовые метрики без ground truth
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy]
)
```

**Цель**: Быстро понять есть ли критические проблемы (галлюцинации, нерелевантные ответы)

### Сценарий 2: Сравнение конфигураций

```python
# Сравните разные значения k
for k in [3, 5, 7, 10]:
    # Evaluate with k documents
    result = evaluate_with_k(k)
    print(f"k={k}: faithfulness={result['faithfulness'].mean():.4f}")
```

**Цель**: Найти оптимальные параметры (k, chunk_size, embedding model)

### Сценарий 3: Полная оценка перед продакшеном

```python
# Используйте все метрики с ground truth
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    ]
)
```

**Цель**: Убедиться что система готова к продакшену

### Сценарий 4: Мониторинг в продакшене

```python
# Логируйте запросы и периодически оценивайте
monitor = RAGMonitor()

# В вашем API
answer, contexts = rag_function(question)
monitor.log_query(question, answer, contexts)

# Раз в неделю
weekly_result = monitor.evaluate_period(start_date, end_date)
```

**Цель**: Отслеживать деградацию качества в продакшене

---

## 🔬 Workflow для оптимизации RAG системы

### Шаг 1: Baseline оценка
```bash
python ragas_eval.py
```
Запишите текущие метрики как baseline.

### Шаг 2: Определите слабые места

Анализируйте метрики:
- **Низкий faithfulness** → Проблема с промптом или моделью
- **Низкий context_precision** → Проблема с retrieval (много шума)
- **Низкий context_recall** → Проблема с retrieval (пропускает информацию)
- **Низкий answer_relevancy** → Проблема с промптом

### Шаг 3: Внесите изменения

Примеры изменений:
- Улучшите промпт
- Измените k (количество документов)
- Попробуйте другую embedding модель
- Измените chunking strategy
- Добавьте re-ranking

### Шаг 4: Переоценка

```python
# Оцените после изменений
improved_result = evaluate(dataset, metrics=[...])

# Сравните с baseline
improvement = improved_result['faithfulness'].mean() - baseline['faithfulness'].mean()
print(f"Улучшение: {improvement:+.4f}")
```

### Шаг 5: Итерация

Повторяйте шаги 2-4 пока не достигнете целевых метрик.

---

## 📈 Интеграция с вашей RAG системой

### Для существующей системы

```python
# 1. Адаптируйте вашу RAG функцию
def your_rag_function(question):
    # Ваш код retrieval
    docs = vector_store.similarity_search(question, k=5)
    contexts = [doc.page_content for doc in docs]
    
    # Ваш код generation
    answer = llm.invoke(create_prompt(question, contexts))
    
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts  # ВАЖНО: список строк!
    }

# 2. Создайте тестовый набор
test_questions = ["Вопрос 1?", "Вопрос 2?", ...]

# 3. Соберите данные
eval_data = [your_rag_function(q) for q in test_questions]

# 4. Оцените
from datasets import Dataset
dataset = Dataset.from_list(eval_data)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
```

### Для нашей системы (с пайплайнами)

```python
# Используйте существующий pipeline_id
pipeline_id = "your-pipeline-id"
variant_id = "your-variant-id"

# Получите конфигурацию
pipeline_data = db_manager.get_pipeline(pipeline_id)
variant_data = db_manager.get_processing_variant(variant_id)

# Создайте RAG функцию
def evaluate_pipeline(pipeline_id, variant_id, test_questions):
    eval_data = []
    
    for question in test_questions:
        # Используйте ваш RAG endpoint
        response = requests.post(
            f"http://localhost:8000/api/rag/query",
            json={
                "pipeline_id": pipeline_id,
                "query": question,
                "llm_provider": "openai",
                "llm_model": "gpt-4o",
                "top_k": 5
            }
        )
        
        data = response.json()
        eval_data.append({
            "question": question,
            "answer": data['answer'],
            "contexts": [s['content'] for s in data['sources']]
        })
    
    return evaluate(Dataset.from_list(eval_data), metrics=[...])
```

---

## 🎓 Обучающий план

### Неделя 1: Основы
- [ ] Прочитать RAGAS_CHEATSHEET.md
- [ ] Запустить ragas_eval.py (Пример 1)
- [ ] Понять метрики faithfulness и answer_relevancy

### Неделя 2: Углубление
- [ ] Прочитать RAGAS_GUIDE.md (разделы 1-4)
- [ ] Создать свой тестовый набор (10-15 вопросов)
- [ ] Запустить оценку на своих данных

### Неделя 3: Практика
- [ ] Прочитать RAGAS_GUIDE.md (разделы 5-7)
- [ ] Создать ground truth для тестовых вопросов
- [ ] Запустить полную оценку (все метрики)
- [ ] Проанализировать результаты

### Неделя 4: Оптимизация
- [ ] Сравнить разные конфигурации (k, embeddings, chunking)
- [ ] Найти оптимальные параметры
- [ ] Настроить мониторинг для продакшена

---

## 💡 Советы и рекомендации

### ✅ DO (Делайте)

1. **Начинайте с малого** - 10-20 вопросов для начала
2. **Используйте temperature=0** для стабильных результатов
3. **Создавайте качественные ground truth** с помощью экспертов
4. **Оценивайте после каждого изменения** для отслеживания прогресса
5. **Документируйте результаты** для сравнения

### ❌ DON'T (Не делайте)

1. **Не полагайтесь на одну метрику** - смотрите на все
2. **Не игнорируйте низкие метрики** - они указывают на проблемы
3. **Не оценивайте слишком часто** - это дорого (LLM вызовы)
4. **Не используйте слишком слабую модель для оценки** - минимум GPT-3.5
5. **Не забывайте про мониторинг** в продакшене

---

## 🔗 Дополнительные ресурсы

### Официальные источники
- 📖 [RAGAS Documentation](https://docs.ragas.io/)
- 🐙 [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- 📝 [RAGAS Paper](https://arxiv.org/abs/2309.15217)

### Наши материалы
- 📘 [RAGAS_GUIDE.md](./RAGAS_GUIDE.md) - Полное руководство
- 📄 [RAGAS_CHEATSHEET.md](./RAGAS_CHEATSHEET.md) - Шпаргалка
- 💻 [ragas_eval.py](./ragas_eval.py) - Примеры кода
- 📊 [test_questions_example.json](./test_questions_example.json) - Тестовые данные

### Связанные документы
- 📚 [TODO.md](./TODO.md) - План развития системы
- 🔧 [REFACTORING.md](./REFACTORING.md) - Архитектура системы
- 🎯 [PIPELINE_SELECTOR_GUIDE.md](./PIPELINE_SELECTOR_GUIDE.md) - Работа с пайплайнами

---

## 📞 Поддержка

Если у вас возникли вопросы:
1. Проверьте [RAGAS_GUIDE.md](./RAGAS_GUIDE.md) раздел Troubleshooting
2. Изучите примеры в [ragas_eval.py](./ragas_eval.py)
3. Посмотрите [официальную документацию](https://docs.ragas.io/)

---

**Последнее обновление**: 2025-12-07  
**Версия**: 1.0

**Удачи в оптимизации вашей RAG системы! 🚀**
