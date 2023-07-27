from typing import List, Dict
from pydantic import BaseModel, Field, ValidationError
import requests
import logging
import tiktoken
import time
import json

class Lloom_config(BaseModel):
    api_key: str
    model: str = "gpt-3.5-turbo"
    temperature: float = Field(0.9, ge=0, le=1)
    max_tokens: int = Field(2000, gt=0)
    top_p: float = Field(1, ge=0, le=1)
    frequency_penalty: float = Field(0, ge=0, le=1)
    presence_penalty: float = Field(0, ge=0, le=1)
    system_message: str = ""
    logging: bool = True

class LLooM:
    token_limits: Dict[str, int] = {
        "gpt-3.5-turbo-0613": 4096,
        "gpt-3.5-turbo-16k-0613": 16384,
        "gpt-4-0314": 8192,
        "gpt-4-32k-0314": 32768,
        "gpt-4-0613": 8192,
        "gpt-4-32k-0613": 32768,
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-0301": 4096,
    }

    def __init__(self, config: lloom_config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.api_key = config.api_key
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.top_p = config.top_p
        self.frequency_penalty = config.frequency_penalty
        self.presence_penalty = config.presence_penalty
        self.system_message: Dict[str, str] = {"role": "system", "content": config.system_message} if config.system_message else None
        self.messages: List[Dict[str, str]] = [self.system_message] if config.system_message else []
        if config.logging:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    def update_config(self, **kwargs):
        current_config_dict = self.config.dict()
        for k, v in kwargs.items():
            if k in current_config_dict:
                current_config_dict[k] = v
            else:
                self.logger.warning(f"Invalid field {k} when trying to update the config, skipping")
        try:
            new_config = GPTConfig(**current_config_dict)
            self.config = new_config
            self.api_key = new_config.api_key
            self.model = new_config.model
            self.temperature = new_config.temperature
            self.max_tokens = new_config.max_tokens
            self.top_p = new_config.top_p
            self.frequency_penalty = new_config.frequency_penalty
            self.presence_penalty = new_config.presence_penalty
            self.system_message: Dict[str, str] = {"role": "system", "content": new_config.system_message} if new_config.system_message else None
            if new_config.logging:
                self.logger.setLevel(logging.INFO)
        except ValidationError as e:
            self.logger.error(f"Invalid value received when trying to update the config, the fields were not updated. {e}")

    def set_system_message(self, content: str):
        self.system_message["content"] = content
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0] = self.system_message
        else:
            self.messages.insert(0, self.system_message)
        self.logger.info(f"Added system message: {content}")

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self.logger.info(f"Added user message: {content}")

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self.logger.info(f"Added assistant message: {content}")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.messages
    
    def clear_history(self, clear_system_message: bool = False) -> List[Dict[str, str]]:
        if not clear_system_message and self.system_message:
            self.messages = [self.system_message]
        else:
            self.messages = []
        return self.messages

    def get_token_count(self, messages: List[Dict[str, str]], model: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.logger.warning(f"Tried counting tokens for {model} but failed. Had to switch to cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4
            tokens_per_name = -1
        elif "gpt-3.5-turbo" in model:
            self.logger.warning("gpt-3.5-turbo may update over time. Returning number of tokens assuming gpt-3.5-turbo-0613.")
            return self.get_token_count(self.messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            self.logger.warning("gpt-4 may update over time. Returning number of tokens tokens assuming gpt-4-0613.")
            return self.get_token_count(self.messages, model="gpt-4-0613")
        else:
            self.logger.warning(
                f'''The function of counting tokens is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens. 
                Your program may fail if you exceed the token limit.'''
            )
            return 0
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens
    
    def validate_token_count(self, prompt_tokens: int) -> bool:
        total_tokens = prompt_tokens + self.max_tokens
        if prompt_tokens > self.token_limits[self.model]:
            self.logger.warning("Token limit is exceeded in the prompt already, deleting the first non-system message")
            if self.messages[0]["role"] == "system":
                del self.messages[1]
            else:
                del self.messages[0]
            prompt_tokens = self.get_token_count(self.messages, self.model)
            return self.validate_token_count(prompt_tokens)
        elif total_tokens > self.token_limits[self.model]:
            self.max_tokens = self.token_limits[self.model] - prompt_tokens
            self.logger.warning(f"Token limit is exceeded, decreased max tokens to {self.max_tokens}")
            return self.validate_token_count(prompt_tokens)
        elif not self.messages:
            self.logger.error("It turns out that after clearing the context, there are no messages at all. Probably you should decrease the length of your message")
            return False
        else:
            return True
        
    def get_completion(self, messages: List[Dict[str, str]]) -> Dict:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "top_p": self.top_p,
            "stop": None
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return(response.json())

    def generate(self, prompt: str) -> str:
        proceed_to_generate = True
        self.add_user_message(prompt)
        if self.model in self.token_limits:
            prompt_tokens = self.get_token_count(self.messages, self.model)
            proceed_to_generate = self.validate_token_count(prompt_tokens)
        else:
            self.logger.warning(f"Unknown model: {self.model}, skipping token counting steps")
        if proceed_to_generate:
            try:
                start_time = time.time()
                self.logger.info(f"Generating for the prompt: {prompt}")
                self.logger.info(f"The current history is: {self.messages}")
                completion = self.get_completion(self.messages)
                response = completion["choices"][0]["message"]['content']
                token_usage = completion["usage"]["total_tokens"]
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.logger.info(f"Consumed {token_usage} tokens, completed in {elapsed_time} seconds")
                self.add_assistant_message(response)
                return response
            except Exception as e:
                self.logger.error(f'An error occurred: {str(e)}, this is the response from OpenAI API: {completion}')
                return str(e)
