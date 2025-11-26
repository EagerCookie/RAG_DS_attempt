from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma

from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

model = ChatOpenAI(model="gpt-4o", temperature=0)

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