import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


class LLMProvider(ABC):
    @abstractmethod
    def get_llm(self) -> Any:
        pass


class EmbeddingProvider(ABC):
    @abstractmethod
    def get_embeddings(self) -> Any:
        pass


class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-pro", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature

    def get_llm(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature
        )


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "models/embedding-001"):
        self.model_name = model_name

    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        return GoogleGenerativeAIEmbeddings(model=self.model_name)


class Config:
    def __init__(self, config_path: str = "config.yml"):
        load_dotenv()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

    def get_llm_provider(self) -> LLMProvider:
        llm_config = self.config.get("llm", {})
        model_name = llm_config.get("model_name", "gemini-pro")
        temperature = llm_config.get("temperature", 0.7)
        return GeminiProvider(model_name, temperature)

    def get_embedding_provider(self) -> EmbeddingProvider:
        embedding_config = self.config.get("embedding", {})
        model_name = embedding_config.get("model_name", "models/embedding-001")
        return GeminiEmbeddingProvider(model_name)

    def get_rag_config(self) -> Dict[str, Any]:
        return self.config.get("rag", {})
