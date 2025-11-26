from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
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
print(f"Количество чанков: {len(all_splits)}")

# Настраиваем rubert-tiny2
# model_name = "cointegrated/rubert-tiny2"
model_name = "DeepVk/USER-bge-m3"
model_kwargs = {'device': 'cpu'} # Используйте 'cuda', если есть GPU
encode_kwargs = {'normalize_embeddings': True} # Важно для косинусного сходства

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

ids = vector_store.add_documents(all_splits)

print(f"Документы добавлены в базу. Размерность вектора: {len(embeddings.embed_query('test'))}")
# Для rubert-tiny2 это будет 312 (у OpenAI было бы 1536 или 3072)