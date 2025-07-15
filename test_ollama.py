
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama

load_dotenv()

ollama_endpoint = os.getenv("OLLAMA_ENDPOINT")

print(f"Connecting to Ollama at: {ollama_endpoint}")

try:
    llm = ChatOllama(base_url=ollama_endpoint, model="gemma:2b")
    response = llm.invoke("Why is the sky blue?")
    print("Successfully connected to Ollama and received a response:")
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")
