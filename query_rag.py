from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma

from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_openai import ChatOpenAI
# from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


# Embedding area
model_name = "DeepVk/USER-bge-m3"
model_kwargs = {'device': 'cpu'} # Используйте 'cuda', если есть GPU
encode_kwargs = {'normalize_embeddings': True} # Важно для косинусного сходства

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    cache_folder="./transformers_models"
)



vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)



# ChatModel Area
model = ChatOpenAI(model="gpt-4o", temperature=0)


# model = ChatDeepSeek(
#     model="deepseek-chat",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,   # other params...
# )

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query, k=5)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are RAG system that should answer user prompt using this information:"
        f"\n\n{docs_content}"
    )
    return system_message

agent = create_agent(model, tools=[], middleware=[prompt_with_context])

query = "Какие типы усилителей используются для измерения биопотенциалов"

response = agent.invoke({"messages": [{"role": "user", "content": query}] })

print(response["messages"][-1].content)