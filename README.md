# Description
This is a simple lightweight wrapper for OpenAI API, created as an alternative to a very complicated and cumbersome Langchain. I was tired of dealing with Langchain and Semantic Kernel abstractions and created my own wrapper which has only necessary features and simplifies the process of with token counting and storing message history. 

# Installation
```bash
pip install lloom
```

# Usage
```python
from lloom import LloomConfig, Lloom

# for OpenAI API only API key is required, all other parameters are optional. Config is managed via Pydantic, therefore you will get an error if some parameters don't adhere to their standards
config = LloomConfig(api_key="sk-...", temperature=0.5)
lloom = Lloom(config)

# method 'generate' will generate a chat completion for you, based on the prompt
response = lloom.generate("Hello!")
```

# Core features: 
- Conversation history: all messages are added to the conversation history, so every time you call generate() it will pass the whole history to make an API call. You can clear the history if you'd like by running the method:
```python
lloom.clear_history() # you can pass the flag clear_system_message=True if you want to delete the system message too
```
- Token counting: tokens are counted automatically for the specified model in config (model="gpt-3.5-turbo" is the default one). The function is almost an identical copy of the script provided by OpenAI in their cookbok repo.
- Token count validation: when the program is done counting tokens, it implements the following logic: if the limit exceeded in the prompt already, it deleted the first user message. If the limit is exceeded yet the prompt is fine, it decreases the max_tokens parameter for completion (default is 2000). If there are no messages left, it will not proceed and you will see an error.
- Logging: every action such as adding a message is logged via logger. You can pass logging=False in config to supress info messages and leave only the warning and error outputs. 

# Available methods: 
- You can update the config at any time:
```python
lloom.update_config(temperature=0.9)
```
- You can provide a system message if you haven't done so in the config:
```python
lloom.set_system_message("You will be doing code review")
```
- You can add user messages and assistant messages in a straightforward fashion, as well as getting the history:
```python
lloom.add_user_messsage("This is a user message")
lloom.add_assistant_message("This is an assistant message")
lloom.get_conversation_history() # returns a list of messages with assigned roles
```
# Azure support
Azure OpenAI API is supported as well: 
```python
from lloom import AzureLloomConfig, AzureLloom
config = AzureLloomConfig(api_key="sk-...", api_base="", api_version="", engine="")
azure_lloom = AzureLloom(config)
```
The main difference is the config which requires api_base, api_version, engine and api_key. All these parameters can be found in your Azure OpenAI studio. All other methods and variables are the same. 

# Notes 
Please try to provide the exact model for the precise token counting. You can omit this parameter or rely on validation, but if it fails, you might get an error from the OpenAI API (you should get a warning via logger). Here is the dictionary with models and their respective limits: 
```python
{
        "gpt-3.5-turbo-0613": 4096,
        "gpt-3.5-turbo-16k-0613": 16384,
        "gpt-4-0314": 8192,
        "gpt-4-32k-0314": 32768,
        "gpt-4-0613": 8192,
        "gpt-4-32k-0613": 32768,
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-0301": 4096,
    }
```
