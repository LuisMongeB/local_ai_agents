import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests

from fastapi import FastAPI, HTTPException, Response


class LLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response from the LLM"""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the LLM service is available"""
        pass


class OllamaClient(LLMClient):
    """Client for Ollama API"""

    def __init__(self, base_url: str = "http://ollama:11434", model: str = "gemma3n"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.generate_url = f"{self.base_url}/api/generate"
        self.tags_url = f"{self.base_url}/api/tags"

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using Ollama API"""
        payload = {
            "prompt": prompt,
            "model": self.model,
            "stream": kwargs.get("stream", False),
            **kwargs,
        }

        try:
            response = requests.post(
                self.generate_url, json=payload, timeout=kwargs.get("timeout", 180)
            )
            response.raise_for_status()
            return Response(response.text, media_type="application/json")
        except requests.exceptions.RequestException as e:
            return {"error": f"Ollama API error: {str(e)}", "success": False}

    def health_check(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        try:
            response = requests.get(self.tags_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to list models: {str(e)}", "success": False}
