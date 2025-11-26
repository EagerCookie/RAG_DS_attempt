from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

file_path = "test.pdf"

loader = PyPDFLoader(file_path)


docs = loader.load()

# print(len(docs))

# print(f"{docs[2].page_content}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = text_splitter.split_documents(docs)
print(len(all_splits))

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

ids = vector_store.add_documents(all_splits)