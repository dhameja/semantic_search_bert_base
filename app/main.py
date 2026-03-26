from fastapi import FastAPI
from app.search import semantic_search


app = FastAPI()


@app.get("/")
def home():
    return {"message": "Semantic Search API Running"}


@app.get("/search")
def search(query: str):
    results = semantic_search(query)
    return {"results": results}