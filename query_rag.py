from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_chroma import Chroma

from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


# Embedding area
model_name = "DeepVk/USER-bge-m3"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    cache_folder="./transformers_models"
)

# ИСПРАВЛЕНИЕ: Укажите полное имя коллекции с суффиксом pipeline_id
vector_store = Chroma(
    collection_name="example_collection_b2be69b0",  # ← ИЗМЕНЕНО!
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db/b2be69b0",
)

# Проверка: выводим количество документов в БД
print(f"📊 Документов в БД: {vector_store._collection.count()}")
print()

# Тест поиска перед использованием в агенте
test_query = "биопотенциалы"
print(f"🔍 Тестовый поиск: '{test_query}'")
test_results = vector_store.similarity_search(test_query, k=3)
print(f"✓ Найдено результатов: {len(test_results)}")
if test_results:
    print(f"   Первый результат: {test_results[0].page_content[:100]}...")
print()

# ChatModel Area
model = ChatOpenAI(model="gpt-4o", temperature=0)


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query, k=5)
    
    # DEBUG: Выводим найденные документы
    print(f"📄 RAG нашел {len(retrieved_docs)} документов для запроса: '{last_query}'")

    if not retrieved_docs:
        print("⚠️  ВНИМАНИЕ: Документы не найдены!")
        docs_content = "Нет релевантной информации в базе знаний."
    else:
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        print(f"✓ Контекст для LLM: {len(docs_content)} символов")

    system_message = (
        "You are RAG system that should answer user prompt using this information:"
        f"\n\n{docs_content}"
    )
    return system_message

agent = create_agent(model, tools=[], middleware=[prompt_with_context])

query = "Какие типы усилителей используются для измерения биопотенциалов"

print("="*60)
print(f"❓ Вопрос: {query}")
print("="*60)
print()

response = agent.invoke({"messages": [{"role": "user", "content": query}] })

print("="*60)
print("💬 Ответ:")
print("="*60)
print(response["messages"][-1].content)