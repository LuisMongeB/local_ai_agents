import requests
from llm_client import OllamaClient

from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Welcome to the FastAPI application!"}


@app.get("/ask")
def ask(prompt: str):
    ollama_client = OllamaClient()
    res = ollama_client.generate(prompt=prompt)

    # Return the dictionary directly - FastAPI will convert it to JSON
    return res
