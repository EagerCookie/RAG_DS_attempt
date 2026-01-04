"""
RAGAS Evaluation Framework - Полное руководство
================================================

Этот файл демонстрирует как использовать RAGAS (RAG Assessment) для оценки качества RAG систем.

RAGAS - это фреймворк для оценки качества Retrieval-Augmented Generation систем.
Он предоставляет набор метрик для измерения различных аспектов качества RAG.

Документация: https://docs.ragas.io/
GitHub: https://github.com/explodinggradients/ragas
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import create_agent

# ============= RAGAS IMPORTS =============
from ragas import evaluate
from ragas.metrics import (
    faithfulness,  # Насколько ответ соответствует контексту (нет галлюцинаций)
    answer_relevancy,  # Релевантность ответа к вопросу
    context_precision,  # Точность извлеченного контекста (нет шума)
    context_recall,  # Полнота извлеченного контекста (все нужное найдено)
    context_entity_recall,  # Полнота сущностей в контексте
    answer_similarity,  # Семантическая схожесть с reference
    answer_correctness  # Корректность ответа (факты + семантика)
)
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

# ============= НАСТРОЙКА EMBEDDINGS =============
# Используем локальную модель для эмбеддингов
model_name = "DeepVk/USER-bge-m3"
model_kwargs = {'device': 'cpu'}  # Используйте 'cuda' если есть GPU
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    cache_folder="./data/transformers_models"
)

# ============= ВЕКТОРНОЕ ХРАНИЛИЩЕ =============
# ВАЖНО: Замените на ваш pipeline_id и collection_name
vector_store = Chroma(
    collection_name="example_collection_b2be69b0",  # ← ИЗМЕНИТЬ на ваш!
    embedding_function=embeddings,
    persist_directory="./data/chroma_langchain_db/b2be69b0",  # ← ИЗМЕНИТЬ на ваш!
)

# ============= CHAT MODEL =============
# LLM для генерации ответов
model = ChatOpenAI(model="gpt-4o", temperature=0, n=1)

# ============= НАСТРОЙКА RAGAS LLM И EMBEDDINGS =============
# RAGAS использует LLM для оценки качества ответов
# Оборачиваем наши модели в RAGAS-совместимые обертки
ragas_llm = LangchainLLMWrapper(model)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# ============= ПРОМПТ С КОНТЕКСТОМ =============
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """
    Динамический промпт который добавляет контекст из векторной БД
    
    Этот middleware:
    1. Извлекает последний вопрос пользователя
    2. Ищет релевантные документы в векторной БД
    3. Добавляет их как контекст в промпт
    4. Сохраняет контекст для последующей оценки RAGAS
    """
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query, k=5)
    
    # Сохраняем контекст для RAGAS - это важно!
    request.state["retrieved_contexts"] = [doc.page_content for doc in retrieved_docs]
    
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    system_message = (
        "You are RAG system that should answer user prompt using this information:"
        f"\n\n{docs_content}"
    )
    return system_message

# Создаем агента с нашим middleware
agent = create_agent(model, tools=[], middleware=[prompt_with_context])


# ============= ФУНКЦИЯ ДЛЯ ГЕНЕРАЦИИ ОТВЕТА =============
def get_rag_response(query: str) -> dict:
    """
    Получить ответ от RAG системы и извлеченный контекст
    
    Args:
        query: Вопрос пользователя
        
    Returns:
        dict с полями:
            - question: исходный вопрос
            - answer: ответ RAG системы
            - contexts: список извлеченных документов (строки)
    
    Примечание:
        RAGAS требует чтобы contexts был списком строк, а не Document объектов!
    """
    # Получаем извлеченный контекст
    retrieved_docs = vector_store.similarity_search(query, k=5)
    contexts = [doc.page_content for doc in retrieved_docs]
    
    # Получаем ответ от LLM
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    answer = response["messages"][-1].content
    
    return {
        "question": query,
        "answer": answer,
        "contexts": contexts  # RAGAS требует список строк
    }


# ============================================================================
# ПРИМЕР 1: БАЗОВАЯ ОЦЕНКА БЕЗ GROUND TRUTH
# ============================================================================
# Этот пример показывает как оценить RAG систему когда у вас НЕТ эталонных ответов.
# Используются метрики которые не требуют ground truth:
# - faithfulness: проверяет нет ли галлюцинаций (ответ основан на контексте)
# - answer_relevancy: проверяет релевантность ответа вопросу

print("=" * 80)
print("ПРИМЕР 1: Базовая оценка (без эталонных ответов)")
print("=" * 80)
print("\nИспользуемые метрики:")
print("  • faithfulness - проверка на галлюцинации")
print("  • answer_relevancy - релевантность ответа вопросу")
print("\nЭти метрики НЕ требуют эталонных ответов!")
print("=" * 80)

# Тестовые вопросы - замените на свои!
test_questions = [
    "Какие типы усилителей используются для измерения биопотенциалов",
    "Что такое операционный усилитель",
    "Какие характеристики имеет идеальный усилитель"
]

# Собираем данные для оценки
eval_data = []
for question in test_questions:
    result = get_rag_response(question)
    eval_data.append(result)
    print(f"\n📝 Вопрос: {question}")
    print(f"💬 Ответ: {result['answer'][:200]}...")
    print(f"📚 Найдено документов: {len(result['contexts'])}")

# Создаем датасет для RAGAS
# Dataset - это специальный формат из библиотеки datasets от HuggingFace
dataset = Dataset.from_list(eval_data)

print("\n🔍 Запуск оценки RAGAS...")
print("⏳ Это может занять некоторое время (LLM оценивает каждый ответ)...")

# Оценка с метриками, не требующими ground truth
result = evaluate(
    dataset,
    metrics=[
        faithfulness,  # Соответствие ответа контексту (0-1, выше = лучше)
        answer_relevancy,  # Релевантность ответа вопросу (0-1, выше = лучше)
    ],
    llm=ragas_llm,  # LLM для оценки (может быть другой моделью)
    embeddings=ragas_embeddings,  # Embeddings для семантического сравнения
)

print("\n" + "=" * 80)
print("📊 РЕЗУЛЬТАТЫ ОЦЕНКИ:")
print("=" * 80)
print(result)
print("\n💡 Интерпретация:")
print("  • Значения от 0 до 1 (чем ближе к 1, тем лучше)")
print("  • faithfulness < 0.7 → возможны галлюцинации")
print("  • answer_relevancy < 0.7 → ответы не по теме")


# ============================================================================
# ПРИМЕР 2: ПОЛНАЯ ОЦЕНКА С GROUND TRUTH
# ============================================================================
# Этот пример показывает как оценить RAG когда у вас ЕСТЬ эталонные ответы.
# Это позволяет использовать дополнительные метрики:
# - context_precision: насколько точен найденный контекст
# - context_recall: насколько полон найденный контекст
# - answer_similarity: семантическая схожесть с эталоном
# - answer_correctness: общая корректность (факты + семантика)

print("\n\n" + "=" * 80)
print("ПРИМЕР 2: Полная оценка (с эталонными ответами)")
print("=" * 80)
print("\nДополнительные метрики:")
print("  • context_precision - точность контекста (нет шума)")
print("  • context_recall - полнота контекста (все нужное найдено)")
print("  • answer_similarity - схожесть с эталоном")
print("  • answer_correctness - общая корректность")
print("\nЭти метрики ТРЕБУЮТ эталонных ответов (ground truth)!")
print("=" * 80)

# Данные с эталонными ответами
# ВАЖНО: Эталонные ответы должны быть написаны экспертами!
test_data_with_ground_truth = [
    {
        "question": "Какие типы усилителей используются для измерения биопотенциалов",
        "ground_truth": "Для измерения биопотенциалов используются инструментальные усилители и операционные усилители с высоким входным сопротивлением"
    },
    {
        "question": "Что такое операционный усилитель",
        "ground_truth": "Операционный усилитель - это интегральная схема с высоким коэффициентом усиления, используемая для усиления сигналов"
    }
]

# Генерируем ответы для вопросов с ground truth
eval_data_full = []
for item in test_data_with_ground_truth:
    result = get_rag_response(item["question"])
    result["ground_truth"] = item["ground_truth"]  # Добавляем эталон
    eval_data_full.append(result)
    print(f"\n📝 Вопрос: {item['question']}")
    print(f"✅ Эталон: {item['ground_truth']}")
    print(f"💬 Ответ RAG: {result['answer'][:150]}...")

# Создаем датасет
dataset_full = Dataset.from_list(eval_data_full)

print("\n🔍 Запуск полной оценки RAGAS...")
print("⏳ Это займет больше времени (больше метрик)...")

# Полная оценка со всеми метриками
result_full = evaluate(
    dataset_full,
    metrics=[
        faithfulness,  # Нет галлюцинаций
        answer_relevancy,  # Релевантность вопросу
        context_precision,  # Точность контекста
        context_recall,  # Полнота контекста
        answer_similarity,  # Схожесть с эталоном
        answer_correctness  # Общая корректность
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

print("\n" + "=" * 80)
print("📊 РЕЗУЛЬТАТЫ ПОЛНОЙ ОЦЕНКИ:")
print("=" * 80)
print(result_full)


# ============================================================================
# ПРИМЕР 3: ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ
# ============================================================================
# Этот пример показывает как анализировать результаты по каждому вопросу

print("\n\n" + "=" * 80)
print("ПРИМЕР 3: Детальный анализ по каждому вопросу")
print("=" * 80)

# Конвертируем результат в pandas DataFrame для удобного анализа
df = result_full.to_pandas()

# Выводим доступные колонки
print("\n📋 Доступные колонки в результате:")
print(df.columns.tolist())

# Выводим таблицу с метриками
print("\n📊 Таблица результатов:")
metric_columns = [col for col in df.columns if col in [
    'faithfulness', 'answer_relevancy', 'context_precision',
    'context_recall', 'answer_similarity', 'answer_correctness'
]]

if metric_columns:
    # Красивый вывод таблицы
    print("\n" + df[['question'] + metric_columns].to_string(index=False))
else:
    print(df)

# Средние значения метрик
print("\n" + "=" * 80)
print("📈 СРЕДНИЕ ЗНАЧЕНИЯ МЕТРИК:")
print("=" * 80)
metrics_summary = {}
for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 
               'context_recall', 'answer_similarity', 'answer_correctness']:
    if metric in df.columns:
        mean_value = df[metric].mean()
        metrics_summary[metric] = mean_value
        
        # Цветная индикация качества
        if mean_value >= 0.8:
            status = "✅ Отлично"
        elif mean_value >= 0.6:
            status = "⚠️  Приемлемо"
        else:
            status = "❌ Требует улучшения"
        
        print(f"{metric:25s}: {mean_value:.4f}  {status}")

# Общая оценка системы
print("\n" + "=" * 80)
print("🎯 ОБЩАЯ ОЦЕНКА СИСТЕМЫ:")
print("=" * 80)
if metrics_summary:
    overall_score = sum(metrics_summary.values()) / len(metrics_summary)
    print(f"Средний балл: {overall_score:.4f}")
    
    if overall_score >= 0.8:
        print("✅ Система работает отлично!")
    elif overall_score >= 0.6:
        print("⚠️  Система работает приемлемо, но есть что улучшить")
    else:
        print("❌ Система требует значительных улучшений")
    
    # Рекомендации
    print("\n💡 Рекомендации по улучшению:")
    if 'faithfulness' in metrics_summary and metrics_summary['faithfulness'] < 0.7:
        print("  • Улучшите промпт чтобы модель строже следовала контексту")
    if 'context_precision' in metrics_summary and metrics_summary['context_precision'] < 0.7:
        print("  • Улучшите retrieval (возможно, уменьшите k или используйте re-ranking)")
    if 'context_recall' in metrics_summary and metrics_summary['context_recall'] < 0.7:
        print("  • Увеличьте k или улучшите chunking стратегию")
    if 'answer_relevancy' in metrics_summary and metrics_summary['answer_relevancy'] < 0.7:
        print("  • Улучшите промпт для более релевантных ответов")


# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ СЦЕНАРИИ ДЛЯ ПРОДВИНУТОГО ИСПОЛЬЗОВАНИЯ
# ============================================================================

# ----------------------------------------------------------------------------
# СЦЕНАРИЙ 1: Batch оценка большого количества вопросов из файла
# ----------------------------------------------------------------------------
def evaluate_batch(questions_file: str):
    """
    Оценка большого батча вопросов из JSON файла
    
    Формат файла:
    [
        {"question": "...", "ground_truth": "..."},
        {"question": "...", "ground_truth": "..."},
        ...
    ]
    
    Args:
        questions_file: путь к JSON файлу с вопросами
        
    Returns:
        результат evaluate()
    """
    import json
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    eval_data = []
    print(f"\n🔄 Обработка {len(questions)} вопросов...")
    
    for i, item in enumerate(questions, 1):
        print(f"  {i}/{len(questions)}: {item['question'][:50]}...")
        result = get_rag_response(item['question'])
        if 'ground_truth' in item:
            result['ground_truth'] = item['ground_truth']
        eval_data.append(result)
    
    dataset = Dataset.from_list(eval_data)
    
    # Выбираем метрики в зависимости от наличия ground truth
    has_ground_truth = 'ground_truth' in questions[0]
    if has_ground_truth:
        metrics = [faithfulness, answer_relevancy, context_precision, 
                  context_recall, answer_similarity, answer_correctness]
    else:
        metrics = [faithfulness, answer_relevancy]
    
    return evaluate(dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_embeddings)


# ----------------------------------------------------------------------------
# СЦЕНАРИЙ 2: Сравнение разных конфигураций RAG (A/B тестирование)
# ----------------------------------------------------------------------------
def compare_rag_configs(test_questions: list):
    """
    Сравнить разные значения k для retrieval
    
    Это позволяет найти оптимальное количество документов для контекста.
    
    Args:
        test_questions: список вопросов для тестирования
        
    Returns:
        dict с результатами для каждого k
    """
    results = {}
    k_values = [3, 5, 7, 10]  # Разные значения k для тестирования
    
    for k in k_values:
        print(f"\n{'='*80}")
        print(f"🔍 Оценка с k={k} документов")
        print(f"{'='*80}")
        
        eval_data = []
        
        for question in test_questions:
            # Получаем k документов
            retrieved_docs = vector_store.similarity_search(question, k=k)
            contexts = [doc.page_content for doc in retrieved_docs]
            
            # Генерируем ответ
            response = agent.invoke({"messages": [{"role": "user", "content": question}]})
            answer = response["messages"][-1].content
            
            eval_data.append({
                "question": question,
                "answer": answer,
                "contexts": contexts
            })
        
        dataset = Dataset.from_list(eval_data)
        result = evaluate(
            dataset, 
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        
        results[f"k={k}"] = result
        
        # Выводим результаты для этого k
        df = result.to_pandas()
        print(f"\n📊 Средние метрики для k={k}:")
        for metric in ['faithfulness', 'answer_relevancy', 'context_precision']:
            if metric in df.columns:
                print(f"  {metric}: {df[metric].mean():.4f}")
    
    # Сравнительная таблица
    print("\n" + "="*80)
    print("📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА:")
    print("="*80)
    print(f"{'k':<10} {'faithfulness':<15} {'answer_relevancy':<20} {'context_precision':<20}")
    print("-"*80)
    
    for k_name, result in results.items():
        df = result.to_pandas()
        f = df['faithfulness'].mean() if 'faithfulness' in df.columns else 0
        ar = df['answer_relevancy'].mean() if 'answer_relevancy' in df.columns else 0
        cp = df['context_precision'].mean() if 'context_precision' in df.columns else 0
        print(f"{k_name:<10} {f:<15.4f} {ar:<20.4f} {cp:<20.4f}")
    
    return results


# ----------------------------------------------------------------------------
# СЦЕНАРИЙ 3: Мониторинг качества в продакшене
# ----------------------------------------------------------------------------
from datetime import datetime

class RAGMonitor:
    """
    Класс для непрерывного мониторинга качества RAG в продакшене
    
    Использование:
        monitor = RAGMonitor()
        
        # В вашем API endpoint:
        answer, contexts = your_rag_function(question)
        monitor.log_query(question, answer, contexts)
        
        # Периодически оценивайте качество:
        results = monitor.evaluate_period(start_date, end_date)
    """
    
    def __init__(self, save_path: str = "./rag_monitoring.json"):
        self.metrics_history = []
        self.save_path = save_path
        self._load_history()
    
    def _load_history(self):
        """Загрузить историю из файла"""
        import json
        import os
        
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Конвертируем timestamp обратно в datetime
                for item in data:
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                self.metrics_history = data
    
    def _save_history(self):
        """Сохранить историю в файл"""
        import json
        
        # Конвертируем datetime в строку для JSON
        data = []
        for item in self.metrics_history:
            item_copy = item.copy()
            item_copy['timestamp'] = item_copy['timestamp'].isoformat()
            data.append(item_copy)
        
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def log_query(self, question: str, answer: str, contexts: list, ground_truth: str = None):
        """
        Логировать запрос для последующей оценки
        
        Args:
            question: вопрос пользователя
            answer: ответ системы
            contexts: список извлеченных документов
            ground_truth: эталонный ответ (опционально)
        """
        entry = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "timestamp": datetime.now()
        }
        
        if ground_truth:
            entry["ground_truth"] = ground_truth
        
        self.metrics_history.append(entry)
        self._save_history()
    
    def evaluate_period(self, start_date: datetime, end_date: datetime):
        """
        Оценить качество за период
        
        Args:
            start_date: начало периода
            end_date: конец периода
            
        Returns:
            результат evaluate()
        """
        # Фильтруем данные за период
        period_data = [
            {k: v for k, v in item.items() if k != "timestamp"}
            for item in self.metrics_history
            if start_date <= item["timestamp"] <= end_date
        ]
        
        if not period_data:
            print("⚠️  Нет данных за указанный период")
            return None
        
        print(f"\n📊 Оценка {len(period_data)} запросов за период:")
        print(f"   От: {start_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   До: {end_date.strftime('%Y-%m-%d %H:%M')}")
        
        dataset = Dataset.from_list(period_data)
        
        # Проверяем наличие ground truth
        has_ground_truth = 'ground_truth' in period_data[0]
        if has_ground_truth:
            metrics = [faithfulness, answer_relevancy, context_precision, 
                      context_recall, answer_similarity, answer_correctness]
        else:
            metrics = [faithfulness, answer_relevancy]
        
        return evaluate(dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_embeddings)
    
    def get_statistics(self):
        """Получить общую статистику"""
        if not self.metrics_history:
            return {"total_queries": 0}
        
        return {
            "total_queries": len(self.metrics_history),
            "first_query": min(item['timestamp'] for item in self.metrics_history),
            "last_query": max(item['timestamp'] for item in self.metrics_history),
            "with_ground_truth": sum(1 for item in self.metrics_history if 'ground_truth' in item)
        }


# ----------------------------------------------------------------------------
# СЦЕНАРИЙ 4: Экспорт результатов в разных форматах
# ----------------------------------------------------------------------------
def export_results(result, filename: str):
    """
    Экспортировать результаты оценки в разных форматах
    
    Args:
        result: результат evaluate()
        filename: базовое имя файла (без расширения)
    """
    import json
    
    # Конвертируем в DataFrame
    df = result.to_pandas()
    
    # 1. Экспорт в CSV
    csv_path = f"{filename}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✅ CSV сохранен: {csv_path}")
    
    # 2. Экспорт в JSON
    json_path = f"{filename}.json"
    result_dict = {
        "scores": result.scores,
        "summary": {
            metric: float(df[metric].mean()) 
            for metric in df.columns 
            if metric in ['faithfulness', 'answer_relevancy', 'context_precision',
                         'context_recall', 'answer_similarity', 'answer_correctness']
        }
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON сохранен: {json_path}")
    
    # 3. Визуализация (если установлен matplotlib)
    try:
        import matplotlib.pyplot as plt
        
        # График средних значений метрик
        metrics = [col for col in df.columns if col in [
            'faithfulness', 'answer_relevancy', 'context_precision',
            'context_recall', 'answer_similarity', 'answer_correctness'
        ]]
        
        if metrics:
            means = [df[m].mean() for m in metrics]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(metrics, means, color=['#667eea', '#764ba2', '#f093fb', 
                                                   '#4facfe', '#00f2fe', '#43e97b'])
            plt.ylim(0, 1)
            plt.ylabel('Score')
            plt.title('RAGAS Metrics Summary')
            plt.xticks(rotation=45, ha='right')
            
            # Добавляем значения на столбцы
            for bar, mean in zip(bars, means):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{mean:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            png_path = f"{filename}.png"
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"✅ График сохранен: {png_path}")
            plt.close()
    except ImportError:
        print("⚠️  matplotlib не установлен, пропускаем визуализацию")


# ============================================================================
# ЗАВЕРШЕНИЕ И СПРАВКА
# ============================================================================

print("\n\n" + "=" * 80)
print("✅ ОЦЕНКА ЗАВЕРШЕНА!")
print("=" * 80)

print("\n📚 СПРАВКА ПО МЕТРИКАМ RAGAS:")
print("-" * 80)
print("""
1. FAITHFULNESS (Верность контексту)
   • Что измеряет: Нет ли в ответе утверждений, не подкрепленных контекстом
   • Как работает: LLM разбивает ответ на факты и проверяет каждый по контексту
   • Диапазон: 0-1 (1 = все факты подтверждены контекстом)
   • Когда низкая: Модель "галлюцинирует" или додумывает информацию
   • Как улучшить: Строже промпт, меньше temperature, лучший контекст

2. ANSWER_RELEVANCY (Релевантность ответа)
   • Что измеряет: Насколько ответ отвечает на вопрос
   • Как работает: Косинусное сходство между вопросом и ответом
   • Диапазон: 0-1 (1 = идеально релевантен)
   • Когда низкая: Ответ не по теме или слишком общий
   • Как улучшить: Улучшить промпт, использовать более умную модель

3. CONTEXT_PRECISION (Точность контекста)
   • Что измеряет: Насколько контекст не зашумлен мусором
   • Как работает: Проверяет релевантность каждого документа к вопросу
   • Диапазон: 0-1 (1 = все документы релевантны)
   • Когда низкая: Retrieval возвращает много нерелевантных документов
   • Как улучшить: Уменьшить k, использовать re-ranking, улучшить chunking

4. CONTEXT_RECALL (Полнота контекста)
   • Что измеряет: Весь ли нужный контекст был найден
   • Как работает: Сравнивает ground truth с извлеченным контекстом
   • Диапазон: 0-1 (1 = весь нужный контекст найден)
   • Когда низкая: Retrieval пропускает важную информацию
   • Как улучшить: Увеличить k, улучшить embeddings, пересмотреть chunking
   • ТРЕБУЕТ: ground_truth

5. ANSWER_SIMILARITY (Схожесть с эталоном)
   • Что измеряет: Семантическая схожесть с эталонным ответом
   • Как работает: Косинусное сходство embeddings ответа и эталона
   • Диапазон: 0-1 (1 = идентичны семантически)
   • Когда низкая: Ответ сильно отличается от эталона
   • Как улучшить: Проверить качество контекста и промпта
   • ТРЕБУЕТ: ground_truth

6. ANSWER_CORRECTNESS (Корректность ответа)
   • Что измеряет: Общая корректность (факты + семантика)
   • Как работает: Комбинация фактической и семантической корректности
   • Диапазон: 0-1 (1 = полностью корректен)
   • Когда низкая: Ответ неправильный или неполный
   • Как улучшить: Улучшить весь pipeline (retrieval + generation)
   • ТРЕБУЕТ: ground_truth
""")

print("\n💡 РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ:")
print("-" * 80)
print("""
1. Начните с базовой оценки (faithfulness + answer_relevancy)
   → Не требует эталонных ответов
   → Быстро показывает основные проблемы

2. Создайте набор эталонных вопросов-ответов
   → 20-50 вопросов покрывающих разные темы
   → Эталоны пишут эксперты предметной области

3. Проводите A/B тестирование конфигураций
   → Разные k (количество документов)
   → Разные chunking стратегии
   → Разные модели embeddings

4. Мониторьте качество в продакшене
   → Логируйте все запросы
   → Периодически оценивайте батчи
   → Отслеживайте деградацию метрик

5. Итеративно улучшайте систему
   → Анализируйте провальные случаи
   → Обновляйте промпты и конфигурацию
   → Переоценивайте после изменений
""")

print("\n🎯 ЦЕЛЕВЫЕ ЗНАЧЕНИЯ МЕТРИК:")
print("-" * 80)
print("""
Отлично (Production-ready):
  • faithfulness: > 0.85
  • answer_relevancy: > 0.80
  • context_precision: > 0.75
  • context_recall: > 0.80

Приемлемо (Требует улучшений):
  • faithfulness: 0.70-0.85
  • answer_relevancy: 0.65-0.80
  • context_precision: 0.60-0.75
  • context_recall: 0.65-0.80

Плохо (Требует серьезной доработки):
  • Любая метрика < 0.60
""")

print("\n" + "=" * 80)
print("📖 Полная документация: https://docs.ragas.io/")
print("=" * 80)