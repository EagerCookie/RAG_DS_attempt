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
    faithfulness,  # Насколько ответ соответствует контексту
    answer_relevancy,  # Релевантность ответа к вопросу
    context_precision,  # Точность извлеченного контекста
    context_recall,  # Полнота извлеченного контекста
    context_entity_recall,  # Полнота сущностей в контексте
    answer_similarity,  # Семантическая схожесть с reference
    answer_correctness  # Корректность ответа
)
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

# ============= НАСТРОЙКА EMBEDDINGS =============
model_name = "DeepVk/USER-bge-m3"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    cache_folder="./transformers_models"
)

# ============= ВЕКТОРНОЕ ХРАНИЛИЩЕ =============
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db/e1dc45d5",
)

# ============= CHAT MODEL =============
model = ChatOpenAI(model="gpt-4o", temperature=0)

# ============= НАСТРОЙКА RAGAS LLM И EMBEDDINGS =============
# Оборачиваем LLM и embeddings для RAGAS
ragas_llm = LangchainLLMWrapper(model)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# ============= ПРОМПТ С КОНТЕКСТОМ =============
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query, k=5)
    
    # Сохраняем контекст для RAGAS
    request.state["retrieved_contexts"] = [doc.page_content for doc in retrieved_docs]
    
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    system_message = (
        "You are RAG system that should answer user prompt using this information:"
        f"\n\n{docs_content}"
    )
    return system_message

agent = create_agent(model, tools=[], middleware=[prompt_with_context])


# ============= ФУНКЦИЯ ДЛЯ ГЕНЕРАЦИИ ОТВЕТА =============
def get_rag_response(query: str):
    """Получить ответ от RAG и извлеченный контекст"""
    # Получаем извлеченный контекст
    retrieved_docs = vector_store.similarity_search(query, k=5)
    contexts = [doc.page_content for doc in retrieved_docs]
    
    # Получаем ответ
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    answer = response["messages"][-1].content
    
    return {
        "question": query,
        "answer": answer,
        "contexts": contexts  # RAGAS требует список строк
    }


# ============= ПРИМЕР 1: БАЗОВАЯ ОЦЕНКА БЕЗ GROUND TRUTH =============
print("=" * 80)
print("ПРИМЕР 1: Базовая оценка (без эталонных ответов)")
print("=" * 80)

# Тестовые вопросы
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
    print(f"\nВопрос: {question}")
    print(f"Ответ: {result['answer'][:200]}...")

# Создаем датасет для RAGAS
dataset = Dataset.from_list(eval_data)

# Оценка с метриками, не требующими ground truth
result = evaluate(
    dataset,
    metrics=[
        faithfulness,  # Соответствие ответа контексту
        answer_relevancy,  # Релевантность ответа
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

print("\n" + "=" * 80)
print("РЕЗУЛЬТАТЫ ОЦЕНКИ:")
print("=" * 80)
print(result)


# ============= ПРИМЕР 2: ПОЛНАЯ ОЦЕНКА С GROUND TRUTH =============
print("\n\n" + "=" * 80)
print("ПРИМЕР 2: Полная оценка (с эталонными ответами)")
print("=" * 80)

# Данные с эталонными ответами для более полной оценки
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
    result["ground_truth"] = item["ground_truth"]
    eval_data_full.append(result)

# Создаем датасет
dataset_full = Dataset.from_list(eval_data_full)

# Полная оценка со всеми метриками
result_full = evaluate(
    dataset_full,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

print("\nРЕЗУЛЬТАТЫ ПОЛНОЙ ОЦЕНКИ:")
print(result_full)


# ============= ПРИМЕР 3: ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ =============
print("\n\n" + "=" * 80)
print("ПРИМЕР 3: Детальный анализ по каждому вопросу")
print("=" * 80)

# Конвертируем результат в pandas для удобного анализа
df = result_full.to_pandas()

# Выводим доступные колонки
print("\nДоступные колонки в результате:")
print(df.columns.tolist())

# Выводим таблицу с доступными метриками
print("\nТаблица результатов:")
metric_columns = [col for col in df.columns if col in [
    'faithfulness', 'answer_relevancy', 'context_precision',
    'context_recall', 'answer_similarity', 'answer_correctness'
]]
if metric_columns:
    print(df[metric_columns])
else:
    print(df)

# Средние значения метрик
print("\n" + "=" * 80)
print("СРЕДНИЕ ЗНАЧЕНИЯ МЕТРИК:")
print("=" * 80)
for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 
               'context_recall', 'answer_similarity', 'answer_correctness']:
    if metric in df.columns:
        print(f"{metric}: {df[metric].mean():.4f}")


# ============= ДОПОЛНИТЕЛЬНЫЕ СЦЕНАРИИ (ЗАКОММЕНТИРОВАНЫ) =============

# # СЦЕНАРИЙ 1: Batch оценка большого количества вопросов
# # -------------------------------------------------------
# # def evaluate_batch(questions_file: str):
# #     """Оценка большого батча вопросов из файла"""
# #     import json
# #     
# #     with open(questions_file, 'r', encoding='utf-8') as f:
# #         questions = json.load(f)
# #     
# #     eval_data = []
# #     for item in questions:
# #         result = get_rag_response(item['question'])
# #         if 'ground_truth' in item:
# #             result['ground_truth'] = item['ground_truth']
# #         eval_data.append(result)
# #     
# #     dataset = Dataset.from_list(eval_data)
# #     return evaluate(dataset, metrics=[faithfulness, answer_relevancy])


# # СЦЕНАРИЙ 2: Сравнение разных конфигураций RAG
# # -----------------------------------------------
# # def compare_rag_configs():
# #     """Сравнить разные k для retrieval"""
# #     results = {}
# #     
# #     for k in [3, 5, 10]:
# #         print(f"\nОценка с k={k}")
# #         eval_data = []
# #         
# #         for question in test_questions:
# #             retrieved_docs = vector_store.similarity_search(question, k=k)
# #             contexts = [doc.page_content for doc in retrieved_docs]
# #             
# #             response = agent.invoke({"messages": [{"role": "user", "content": question}]})
# #             answer = response["messages"][-1].content
# #             
# #             eval_data.append({
# #                 "question": question,
# #                 "answer": answer,
# #                 "contexts": contexts  # Список строк
# #             })
# #         
# #         dataset = Dataset.from_list(eval_data)
# #         results[f"k={k}"] = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
# #     
# #     return results


# # СЦЕНАРИЙ 3: Мониторинг качества в продакшене
# # ----------------------------------------------
# # from datetime import datetime
# # 
# # class RAGMonitor:
# #     """Класс для непрерывного мониторинга качества RAG"""
# #     
# #     def __init__(self):
# #         self.metrics_history = []
# #     
# #     def log_query(self, question: str, answer: str, contexts: list):
# #         """Логировать запрос для последующей оценки"""
# #         self.metrics_history.append({
# #             "question": question,
# #             "answer": answer,
# #             "contexts": contexts,  # Список строк
# #             "timestamp": datetime.now()
# #         })
# #     
# #     def evaluate_period(self, start_date, end_date):
# #         """Оценить качество за период"""
# #         period_data = [
# #             {k: v for k, v in item.items() if k != "timestamp"}
# #             for item in self.metrics_history
# #             if start_date <= item["timestamp"] <= end_date
# #         ]
# #         
# #         dataset = Dataset.from_list(period_data)
# #         return evaluate(dataset, metrics=[faithfulness, answer_relevancy])


# # СЦЕНАРИЙ 4: A/B тестирование промптов
# # --------------------------------------
# # def ab_test_prompts():
# #     """Сравнить разные варианты промптов"""
# #     
# #     prompts = {
# #         "simple": "Answer based on: {context}",
# #         "detailed": "You are RAG system. Use this info: {context}",
# #         "structured": "Context:\n{context}\n\nProvide detailed answer:"
# #     }
# #     
# #     results = {}
# #     for name, prompt_template in prompts.items():
# #         # Создать агента с новым промптом
# #         # Оценить на тестовых данных
# #         # Сохранить результаты
# #         pass
# #     
# #     return results


# # СЦЕНАРИЙ 5: Оценка с использованием LLM-as-judge
# # -------------------------------------------------
# # from ragas.llms import LangchainLLMWrapper
# # 
# # # Использовать другую модель для оценки
# # judge_model = ChatOpenAI(model="gpt-4-turbo")
# # judge_llm = LangchainLLMWrapper(judge_model)
# # 
# # result = evaluate(
# #     dataset,
# #     metrics=[faithfulness, answer_relevancy],
# #     llm=judge_llm,
# #     embeddings=ragas_embeddings
# # )


# # СЦЕНАРИЙ 6: Кастомные метрики
# # ------------------------------
# # from ragas.metrics.base import Metric
# # 
# # class CustomRelevanceMetric(Metric):
# #     """Своя метрика релевантности"""
# #     
# #     def __init__(self):
# #         self.name = "custom_relevance"
# #     
# #     def score(self, question: str, answer: str, contexts: list) -> float:
# #         # Ваша логика оценки
# #         return 0.85


# # СЦЕНАРИЙ 7: Генерация синтетических тестовых данных
# # ----------------------------------------------------
# # from ragas.testset.generator import TestsetGenerator
# # from ragas.testset.evolutions import simple, reasoning, multi_context
# # 
# # # Загрузить документы
# # documents = [doc.page_content for doc in vector_store.get()]
# # 
# # # Создать генератор
# # generator = TestsetGenerator.with_openai()
# # 
# # # Сгенерировать тестовый набор
# # testset = generator.generate_with_langchain_docs(
# #     documents,
# #     test_size=10,
# #     distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}
# # )


# # СЦЕНАРИЙ 8: Экспорт результатов
# # --------------------------------
# # def export_results(result, filename: str):
# #     """Экспортировать результаты в разных форматах"""
# #     
# #     # В CSV
# #     df = result.to_pandas()
# #     df.to_csv(f"{filename}.csv", index=False)
# #     
# #     # В JSON
# #     import json
# #     with open(f"{filename}.json", 'w', encoding='utf-8') as f:
# #         json.dump(result.scores, f, ensure_ascii=False, indent=2)
# #     
# #     # Визуализация
# #     import matplotlib.pyplot as plt
# #     df[['faithfulness', 'answer_relevancy']].plot(kind='bar')
# #     plt.savefig(f"{filename}.png")


print("\n\n" + "=" * 80)
print("ОЦЕНКА ЗАВЕРШЕНА!")
print("=" * 80)
print("\nОбъяснение метрик:")
print("- faithfulness: показывает, насколько ответ основан на предоставленном контексте (0-1)")
print("- answer_relevancy: насколько ответ релевантен вопросу (0-1)")
print("- context_precision: точность извлеченного контекста (0-1)")
print("- context_recall: полнота извлеченного контекста (0-1)")
print("- answer_similarity: семантическая схожесть с эталонным ответом (0-1)")
print("- answer_correctness: общая корректность ответа (0-1)")