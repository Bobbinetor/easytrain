"""Ollama client for LLM interactions"""

import requests
import json
import logging
from typing import Optional, Dict, Any
from .config import OllamaConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.base_url = config.host.rstrip('/')
        
    def is_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Ollama server not available: {e}")
            return False
    
    def list_models(self) -> list:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.RequestException as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a specific model exists"""
        models = self.list_models()
        return any(model.get("name") == model_name for model in models)
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None,
                model: Optional[str] = None,
                stream: bool = False,
                **kwargs) -> Dict[str, Any]:
        """Generate response using Ollama"""
        
        model_name = model or self.config.model
        
        # Check if model exists
        if not self.model_exists(model_name):
            available_models = [m.get("name") for m in self.list_models()]
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            logger.info(f"Sending request to Ollama with model: {model_name}")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                result = response.json()
                logger.info(f"Received response from Ollama: {len(result.get('response', ''))} characters")
                return result
                
        except requests.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise
    
    def _handle_streaming_response(self, response) -> Dict[str, Any]:
        """Handle streaming response from Ollama"""
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if chunk.get("response"):
                        full_response += chunk["response"]
                    if chunk.get("done"):
                        return {
                            "response": full_response,
                            "done": True
                        }
                except json.JSONDecodeError:
                    continue
        
        return {"response": full_response, "done": True}
    
    def chat(self, messages: list, model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Chat completion using Ollama"""
        model_name = model or self.config.model
        
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Error in chat completion: {e}")
            raise