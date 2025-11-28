from fastapi import FastAPI
from app.api import loaders, splitters, embeddings, vectorstores, pipelines

app = FastAPI()

app.include_router(loaders.router)
app.include_router(splitters.router)
app.include_router(embeddings.router)
app.include_router(vectorstores.router)
app.include_router(pipelines.router)
