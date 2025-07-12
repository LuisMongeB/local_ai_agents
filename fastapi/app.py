import requests

from fastapi import FastAPI, Response

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Welcome to the FastAPI application!"}


@app.get("/ask")
def ask(prompt: str):
    res = requests.post(
        "http://ollama:11434/api/generate",
        json={"prompt": prompt, "stream": False, "model": "gemma3n"},
    )

    return Response(content=res.text, media_type="application/json")
