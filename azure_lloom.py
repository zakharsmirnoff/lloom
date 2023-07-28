from typing import List, Dict, Optional
from pydantic import BaseModel, Field, ValidationError
import logging
from lloom import LLooM
import requests
import json

class AzureLloom(BaseModel):
    api_key: str
    api_base: str
    api_version: str
    model: str = "gpt-3.5-turbo-0301"
    engine: str
    temperature: float = Field(0.9, ge=0, le=1)
    max_tokens: int = Field(2000, gt=0)
    top_p: float = Field(1, ge=0, le=1)
    frequency_penalty: float = Field(0, ge=0, le=1)
    presence_penalty: float = Field(0, ge=0, le=1)
    system_message: str = ""
    logging: bool = True

class AzureLLooM(LLooM):
    def __init__(self, config: AzureLloom):
        self.config = config
        self.logger = logging.getLogger(__name__)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.api_type = "azure"
        self.api_key = config.api_key
        self.api_base = config.api_base
        self.api_version = config.api_version
        self.model = config.model
        self.engine = config.engine
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.top_p = config.top_p
        self.frequency_penalty = config.frequency_penalty
        self.presence_penalty = config.presence_penalty
        self.system_message: Dict[str, str] = {"role": "system", "content": config.system_message}
        self.messages: List[Dict[str, str]] = [self.system_message] if config.system_message else []
        if config.logging:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    def get_completion(self):
        url = f"{self.api_base}openai/deployments/{self.engine}/chat/completions?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        data = {
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "top_p": self.top_p,
            "stop": None
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return(response.json())
