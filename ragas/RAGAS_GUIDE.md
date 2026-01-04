# RAGAS Framework - Полное руководство для оценки RAG систем

## 📚 Содержание

1. [Введение в RAGAS](#введение)
2. [Установка и настройка](#установка)
3. [Основные концепции](#концепции)
4. [Метрики RAGAS](#метрики)
5. [Практические примеры](#примеры)
6. [Сравнение конфигураций пайплайнов](#сравнение)
7. [Интерпретация результатов](#интерпретация)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Введение в RAGAS {#введение}

**RAGAS** (RAG Assessment) - это фреймворк для автоматической оценки качества RAG (Retrieval-Augmented Generation) систем.

### Зачем нужен RAGAS?

При разработке RAG системы возникают вопросы:
- ❓ Насколько хорошо работает мой retrieval?
- ❓ Не "галлюцинирует" ли модель?
- ❓ Какая конфигурация лучше: k=5 или k=10?
- ❓ Какой chunking strategy использовать?

**RAGAS дает объективные метрики** для ответа на эти вопросы.

### Ключевые преимущества

✅ **Автоматическая оценка** - не нужно вручную проверять каждый ответ  
✅ **Множество метрик** - оценка разных аспектов качества  
✅ **LLM-as-judge** - использует LLM для оценки качества  
✅ **Интеграция с LangChain** - легко подключить к существующей системе  
✅ **A/B тестирование** - сравнение разных конфигураций  

---

## 🔧 Установка и настройка {#установка}

### Установка зависимостей

```bash
pip install ragas datasets langchain langchain-openai langchain-huggingface
```

### Необходимые API ключи

RAGAS использует LLM для оценки, поэтому нужен API ключ:

```bash
# .env файл
OPENAI_API_KEY=sk-...
# или
ANTHROPIC_API_KEY=sk-ant-...
```

### Базовая настройка

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# LLM для оценки (может отличаться от LLM для генерации)
evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
ragas_llm = LangchainLLMWrapper(evaluator_llm)
```

💡 **Совет**: Для оценки можно использовать более дешевую модель (gpt-4o-mini), чем для генерации ответов.

---

## 🧠 Основные концепции {#концепции}

### Структура данных для RAGAS

RAGAS требует данные в специальном формате:

```python
from datasets import Dataset

# Минимальный формат (без ground truth)
data = [
    {
        "question": "Что такое RAG?",
        "answer": "RAG - это Retrieval-Augmented Generation...",
        "contexts": [
            "RAG combines retrieval and generation...",
            "The retrieval component finds relevant docs..."
        ]
    }
]

# Полный формат (с ground truth)
data_full = [
    {
        "question": "Что такое RAG?",
        "answer": "RAG - это Retrieval-Augmented Generation...",
        "contexts": ["...", "..."],
        "ground_truth": "RAG - это метод который комбинирует..."  # Эталон
    }
]

dataset = Dataset.from_list(data)
```

### Типы метрик

RAGAS метрики делятся на две категории:

#### 1. **Метрики БЕЗ ground truth** (не требуют эталонных ответов)
- `faithfulness` - проверка на галлюцинации
- `answer_relevancy` - релевантность ответа вопросу

#### 2. **Метрики С ground truth** (требуют эталонные ответы)
- `context_precision` - точность retrieval
- `context_recall` - полнота retrieval
- `answer_similarity` - схожесть с эталоном
- `answer_correctness` - общая корректность

---

## 📊 Метрики RAGAS {#метрики}

### 1. Faithfulness (Верность контексту)

**Что измеряет**: Нет ли в ответе утверждений, не подкрепленных контекстом (галлюцинации).

**Как работает**:
1. LLM разбивает ответ на отдельные утверждения (claims)
2. Для каждого утверждения проверяет: есть ли оно в контексте?
3. Считает долю подтвержденных утверждений

**Пример**:

```python
# Контекст
contexts = ["Операционный усилитель имеет высокий коэффициент усиления"]

# Ответ
answer = "Операционный усилитель имеет высокий коэффициент усиления и низкое энергопотребление"

# Утверждения:
# 1. "Операционный усилитель имеет высокий коэффициент усиления" ✅ (есть в контексте)
# 2. "Операционный усилитель имеет низкое энергопотребление" ❌ (нет в контексте)

# Faithfulness = 1/2 = 0.5
```

**Интерпретация**:
- `1.0` - идеально, все утверждения подтверждены
- `0.8-1.0` - отлично
- `0.6-0.8` - приемлемо
- `< 0.6` - много галлюцинаций

**Как улучшить**:
- Более строгий промпт ("Answer ONLY based on context")
- Уменьшить temperature модели
- Использовать более качественный контекст

---

### 2. Answer Relevancy (Релевантность ответа)

**Что измеряет**: Насколько ответ релевантен вопросу.

**Как работает**:
1. Вычисляет embeddings вопроса
2. Вычисляет embeddings ответа
3. Считает косинусное сходство

**Пример**:

```python
# Вопрос
question = "Какие типы усилителей используются?"

# Хороший ответ (релевантный)
answer = "Используются операционные и инструментальные усилители"
# relevancy ≈ 0.9

# Плохой ответ (нерелевантный)
answer = "Усилители - это важные компоненты электроники"
# relevancy ≈ 0.4
```

**Интерпретация**:
- `0.9-1.0` - ответ точно по теме
- `0.7-0.9` - ответ релевантен, но может быть лучше
- `< 0.7` - ответ не по теме или слишком общий

**Как улучшить**:
- Улучшить промпт (добавить "Be specific and direct")
- Использовать более умную модель
- Добавить примеры в промпт (few-shot)

---

### 3. Context Precision (Точность контекста)

**Что измеряет**: Насколько извлеченный контекст не зашумлен нерелевантными документами.

**Как работает**:
1. Для каждого документа в contexts проверяет: релевантен ли он вопросу?
2. Считает precision = (релевантные документы) / (все документы)

**Пример**:

```python
# Вопрос
question = "Что такое операционный усилитель?"

# Контекст (5 документов)
contexts = [
    "Операционный усилитель - это интегральная схема...",  # ✅ релевантен
    "Усилители используются в аудио системах...",          # ✅ релевантен
    "История развития электроники началась...",            # ❌ нерелевантен
    "Операционные усилители имеют высокое усиление...",    # ✅ релевантен
    "Резисторы - это пассивные компоненты..."              # ❌ нерелевантен
]

# Precision = 3/5 = 0.6
```

**Интерпретация**:
- `0.8-1.0` - отличный retrieval, мало шума
- `0.6-0.8` - приемлемо, но есть лишние документы
- `< 0.6` - много нерелевантных документов

**Как улучшить**:
- Уменьшить k (количество документов)
- Использовать re-ranking
- Улучшить embeddings модель
- Пересмотреть chunking strategy

---

### 4. Context Recall (Полнота контекста)

**Что измеряет**: Весь ли необходимый контекст был найден retrieval системой.

**Как работает**:
1. Сравнивает ground_truth с извлеченным контекстом
2. Проверяет: вся ли информация из ground_truth есть в contexts?
3. Считает recall = (найденная информация) / (вся нужная информация)

**Пример**:

```python
# Ground truth (эталонный ответ)
ground_truth = "Операционные усилители имеют высокое усиление и высокое входное сопротивление"

# Извлеченный контекст
contexts = [
    "Операционные усилители характеризуются высоким коэффициентом усиления",  # ✅ есть "высокое усиление"
    "Резисторы используются в схемах"  # ❌ нет информации о входном сопротивлении
]

# Recall ≈ 0.5 (найдена только половина информации)
```

**Интерпретация**:
- `0.9-1.0` - вся нужная информация найдена
- `0.7-0.9` - большая часть найдена
- `< 0.7` - много пропущенной информации

**Как улучшить**:
- Увеличить k (больше документов)
- Улучшить embeddings
- Пересмотреть chunking (возможно, чанки слишком маленькие)
- Проверить качество индексации

---

### 5. Answer Similarity (Схожесть с эталоном)

**Что измеряет**: Семантическая схожесть ответа с эталонным ответом.

**Как работает**:
1. Вычисляет embeddings ответа системы
2. Вычисляет embeddings ground_truth
3. Считает косинусное сходство

**Пример**:

```python
# Ground truth
ground_truth = "Операционный усилитель - это интегральная схема с высоким усилением"

# Ответ системы (семантически похож)
answer = "ОУ представляет собой ИС с большим коэффициентом усиления"
# similarity ≈ 0.95

# Ответ системы (семантически отличается)
answer = "Усилители используются для обработки сигналов"
# similarity ≈ 0.4
```

**Интерпретация**:
- `0.9-1.0` - ответ очень близок к эталону
- `0.7-0.9` - ответ правильный, но формулировка отличается
- `< 0.7` - ответ сильно отличается

💡 **Важно**: Низкий similarity не всегда плохо - ответ может быть правильным, но сформулирован иначе.

---

### 6. Answer Correctness (Корректность ответа)

**Что измеряет**: Общая корректность ответа (комбинация фактов и семантики).

**Как работает**:
1. Извлекает факты из ответа и ground_truth
2. Сравнивает факты (фактическая корректность)
3. Сравнивает семантику (семантическая корректность)
4. Комбинирует оба аспекта

**Формула**:
```
correctness = w1 * factual_correctness + w2 * semantic_similarity
```

**Интерпретация**:
- `0.9-1.0` - ответ полностью корректен
- `0.7-0.9` - ответ в основном правильный
- `< 0.7` - ответ содержит ошибки

**Как улучшить**:
- Улучшить весь pipeline (retrieval + generation)
- Проверить качество ground_truth
- Использовать более мощную модель

---

## 💻 Практические примеры {#примеры}

### Пример 1: Быстрая оценка без ground truth

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# Ваши данные
data = [
    {
        "question": "Что такое RAG?",
        "answer": "RAG - это метод который комбинирует retrieval и generation",
        "contexts": ["RAG combines retrieval with generation..."]
    }
]

dataset = Dataset.from_list(data)

# Оценка
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

print(result)
```

### Пример 2: Полная оценка с ground truth

```python
# Данные с эталонами
data = [
    {
        "question": "Что такое RAG?",
        "answer": "RAG - это метод который комбинирует retrieval и generation",
        "contexts": ["RAG combines retrieval with generation..."],
        "ground_truth": "RAG (Retrieval-Augmented Generation) - это подход..."
    }
]

dataset = Dataset.from_list(data)

# Полная оценка
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

# Анализ результатов
df = result.to_pandas()
print("\nСредние значения:")
for metric in df.columns:
    if metric.startswith(('faithfulness', 'answer', 'context')):
        print(f"{metric}: {df[metric].mean():.4f}")
```

### Пример 3: Batch оценка из файла

```python
import json
from datasets import Dataset

# Загрузить вопросы из JSON
with open('test_questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

# Формат файла:
# [
#     {"question": "...", "ground_truth": "..."},
#     {"question": "...", "ground_truth": "..."}
# ]

# Генерировать ответы
eval_data = []
for item in questions:
    # Ваша RAG функция
    answer, contexts = your_rag_function(item['question'])
    
    eval_data.append({
        "question": item['question'],
        "answer": answer,
        "contexts": contexts,
        "ground_truth": item.get('ground_truth')
    })

dataset = Dataset.from_list(eval_data)
result = evaluate(dataset, metrics=[...])
```

---

## 🔬 Сравнение конфигураций пайплайнов {#сравнение}

### Сценарий: Найти оптимальное значение k

```python
def compare_k_values(test_questions, k_values=[3, 5, 7, 10]):
    """Сравнить разные значения k для retrieval"""
    
    results = {}
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Тестирование k={k}")
        print('='*60)
        
        eval_data = []
        for question in test_questions:
            # Retrieval с текущим k
            docs = vector_store.similarity_search(question, k=k)
            contexts = [doc.page_content for doc in docs]
            
            # Генерация ответа
            answer = llm.invoke(create_prompt(question, contexts))
            
            eval_data.append({
                "question": question,
                "answer": answer,
                "contexts": contexts
            })
        
        dataset = Dataset.from_list(eval_data)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision]
        )
        
        results[f"k={k}"] = result.to_pandas()
    
    # Сравнительная таблица
    print("\n" + "="*80)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print("="*80)
    
    comparison = {}
    for k_name, df in results.items():
        comparison[k_name] = {
            'faithfulness': df['faithfulness'].mean(),
            'answer_relevancy': df['answer_relevancy'].mean(),
            'context_precision': df['context_precision'].mean()
        }
    
    import pandas as pd
    comparison_df = pd.DataFrame(comparison).T
    print(comparison_df)
    
    # Найти лучшее k
    best_k = comparison_df['faithfulness'].idxmax()
    print(f"\n🏆 Лучшее k: {best_k}")
    
    return comparison_df
```

### Сценарий: Сравнение разных embedding моделей

```python
def compare_embeddings(test_questions):
    """Сравнить разные модели embeddings"""
    
    embedding_models = [
        "cointegrated/rubert-tiny2",
        "DeepVk/USER-bge-m3",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    results = {}
    
    for model_name in embedding_models:
        print(f"\nТестирование: {model_name}")
        
        # Создать embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder="./transformers_models"
        )
        
        # Создать vector store
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=f"./test_db_{model_name.replace('/', '_')}"
        )
        
        # Оценить
        # ... (аналогично предыдущему примеру)
        
    return results
```

### Сценарий: Сравнение chunking strategies

```python
def compare_chunking_strategies(documents):
    """Сравнить разные стратегии разбиения на чанки"""
    
    strategies = {
        "small": {"chunk_size": 256, "chunk_overlap": 50},
        "medium": {"chunk_size": 512, "chunk_overlap": 100},
        "large": {"chunk_size": 1024, "chunk_overlap": 200}
    }
    
    results = {}
    
    for name, params in strategies.items():
        print(f"\nТестирование: {name} chunks")
        
        # Создать splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=params['chunk_size'],
            chunk_overlap=params['chunk_overlap']
        )
        
        # Разбить документы
        chunks = splitter.split_documents(documents)
        
        # Создать vector store
        vector_store = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=f"./test_db_{name}"
        )
        
        # Оценить
        # ... (аналогично предыдущему примеру)
        
    return results
```

---

## 📈 Интерпретация результатов {#интерпретация}

### Таблица целевых значений

| Метрика | Отлично | Хорошо | Приемлемо | Плохо |
|---------|---------|--------|-----------|-------|
| **faithfulness** | > 0.85 | 0.75-0.85 | 0.60-0.75 | < 0.60 |
| **answer_relevancy** | > 0.80 | 0.70-0.80 | 0.55-0.70 | < 0.55 |
| **context_precision** | > 0.75 | 0.65-0.75 | 0.50-0.65 | < 0.50 |
| **context_recall** | > 0.80 | 0.70-0.80 | 0.55-0.70 | < 0.55 |
| **answer_similarity** | > 0.85 | 0.75-0.85 | 0.60-0.75 | < 0.60 |
| **answer_correctness** | > 0.80 | 0.70-0.80 | 0.55-0.70 | < 0.55 |

### Диагностика проблем

#### Низкий faithfulness (< 0.7)
**Проблема**: Модель "галлюцинирует"

**Решения**:
1. Строже промпт: "Answer ONLY based on the provided context"
2. Уменьшить temperature до 0
3. Использовать более надежную модель
4. Добавить в промпт: "If you don't know, say 'I don't know'"

#### Низкий context_precision (< 0.6)
**Проблема**: Retrieval возвращает много мусора

**Решения**:
1. Уменьшить k (меньше документов)
2. Использовать re-ranking (Cohere, Cross-Encoder)
3. Улучшить chunking (меньше размер чанков)
4. Использовать лучшую embedding модель

#### Низкий context_recall (< 0.6)
**Проблема**: Retrieval пропускает важную информацию

**Решения**:
1. Увеличить k (больше документов)
2. Улучшить chunking (больше overlap)
3. Использовать hybrid search (keyword + semantic)
4. Проверить качество индексации

#### Низкий answer_relevancy (< 0.6)
**Проблема**: Ответы не по теме

**Решения**:
1. Улучшить промпт (добавить примеры)
2. Использовать более умную модель
3. Добавить chain-of-thought reasoning
4. Проверить качество контекста

---

## ✅ Best Practices {#best-practices}

### 1. Создание тестового набора

**Рекомендации**:
- 📝 Минимум 20-30 вопросов для начала
- 🎯 Покрывайте разные темы и сложности
- 👥 Эталоны пишут эксперты предметной области
- 🔄 Регулярно обновляйте тестовый набор

**Структура тестового набора**:

```json
[
  {
    "question": "Простой фактический вопрос",
    "ground_truth": "Прямой ответ",
    "category": "factual",
    "difficulty": "easy"
  },
  {
    "question": "Вопрос требующий рассуждений",
    "ground_truth": "Развернутый ответ с объяснением",
    "category": "reasoning",
    "difficulty": "medium"
  },
  {
    "question": "Вопрос требующий информации из нескольких источников",
    "ground_truth": "Ответ синтезирующий информацию",
    "category": "multi-context",
    "difficulty": "hard"
  }
]
```

### 2. Итеративное улучшение

**Процесс**:
1. 📊 Оценить текущую систему
2. 🔍 Найти слабые места (низкие метрики)
3. 🛠️ Внести изменения
4. 📊 Переоценить
5. 🔄 Повторить

**Пример workflow**:

```python
# 1. Baseline оценка
baseline_result = evaluate(dataset, metrics=[...])
print(f"Baseline faithfulness: {baseline_result['faithfulness'].mean():.4f}")

# 2. Изменение (например, улучшение промпта)
# ... modify your system ...

# 3. Новая оценка
improved_result = evaluate(dataset, metrics=[...])
print(f"Improved faithfulness: {improved_result['faithfulness'].mean():.4f}")

# 4. Сравнение
improvement = improved_result['faithfulness'].mean() - baseline_result['faithfulness'].mean()
print(f"Improvement: {improvement:+.4f}")
```

### 3. Мониторинг в продакшене

**Рекомендации**:
- 📝 Логируйте все запросы (question, answer, contexts)
- ⏰ Оценивайте батчи периодически (раз в неделю/месяц)
- 📉 Отслеживайте деградацию метрик
- 🚨 Настройте алерты на критические падения

**Пример**:

```python
from datetime import datetime, timedelta

# Мониторинг за последнюю неделю
monitor = RAGMonitor()

# Оценка
end_date = datetime.now()
start_date = end_date - timedelta(days=7)
weekly_result = monitor.evaluate_period(start_date, end_date)

# Проверка на деградацию
if weekly_result['faithfulness'].mean() < 0.7:
    send_alert("RAG quality degradation detected!")
```

### 4. Выбор LLM для оценки

**Рекомендации**:
- 💰 Для оценки можно использовать более дешевую модель
- 🎯 Но не слишком слабую (минимум GPT-3.5 / Claude Haiku)
- 🔄 Периодически сравнивайте с более мощной моделью

**Пример**:

```python
# Для регулярной оценки (дешево)
cheap_llm = ChatOpenAI(model="gpt-4o-mini")

# Для важных оценок (точнее)
premium_llm = ChatOpenAI(model="gpt-4o")

# Периодическая проверка
if datetime.now().day == 1:  # Раз в месяц
    result = evaluate(dataset, llm=LangchainLLMWrapper(premium_llm))
else:
    result = evaluate(dataset, llm=LangchainLLMWrapper(cheap_llm))
```

---

## 🔧 Troubleshooting {#troubleshooting}

### Проблема: "RateLimitError" при оценке

**Причина**: Слишком много запросов к API

**Решение**:
```python
import time

# Добавить задержки между запросами
for item in eval_data:
    result = get_rag_response(item['question'])
    eval_data.append(result)
    time.sleep(1)  # Пауза 1 секунда
```

### Проблема: Оценка занимает слишком много времени

**Причина**: RAGAS делает много LLM вызовов

**Решения**:
1. Уменьшить размер тестового набора
2. Использовать более быструю модель для оценки
3. Оценивать только критические метрики
4. Использовать батчинг

```python
# Оценивать только важные метрики
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],  # Только 2 метрики
    llm=ragas_llm
)
```

### Проблема: Метрики нестабильны (сильно варьируются)

**Причина**: Малый размер тестового набора или стохастичность LLM

**Решения**:
1. Увеличить тестовый набор (минимум 30 вопросов)
2. Использовать temperature=0 для оценки
3. Запускать оценку несколько раз и усреднять

```python
# Множественные прогоны
results = []
for _ in range(3):
    result = evaluate(dataset, metrics=[...])
    results.append(result.to_pandas())

# Усреднение
avg_faithfulness = sum(r['faithfulness'].mean() for r in results) / 3
```

### Проблема: Низкие метрики даже для хорошей системы

**Причина**: Плохие ground truth или несоответствие стиля

**Решения**:
1. Проверить качество ground truth
2. Убедиться что ground truth написаны в том же стиле что и ответы
3. Использовать метрики без ground truth для проверки

```python
# Проверка: оценить ground truth как ответы
test_data = [
    {
        "question": q['question'],
        "answer": q['ground_truth'],  # Используем ground truth как ответ
        "contexts": q['contexts']
    }
    for q in eval_data
]

# Должно быть близко к 1.0
result = evaluate(Dataset.from_list(test_data), metrics=[answer_similarity])
```

---

## 📚 Дополнительные ресурсы

- 📖 [Официальная документация RAGAS](https://docs.ragas.io/)
- 🎓 [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- 📝 [Статья о RAGAS](https://arxiv.org/abs/2309.15217)
- 🎥 [Видео туториалы](https://www.youtube.com/results?search_query=ragas+evaluation)

---

## 🎯 Чек-лист для начала работы

- [ ] Установить RAGAS и зависимости
- [ ] Настроить API ключи (OpenAI/Anthropic)
- [ ] Создать тестовый набор (минимум 20 вопросов)
- [ ] Запустить базовую оценку (faithfulness + answer_relevancy)
- [ ] Создать ground truth для части вопросов
- [ ] Запустить полную оценку
- [ ] Проанализировать результаты
- [ ] Определить слабые места
- [ ] Внести улучшения
- [ ] Переоценить и сравнить
- [ ] Настроить мониторинг для продакшена

---

**Последнее обновление**: 2025-12-07  
**Версия**: 1.0
