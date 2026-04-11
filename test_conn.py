import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

# Validar OpenAI
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-08-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

try:
    res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":"Hola"}])
    print("✅ OpenAI: Conectado")
except Exception as e:
    print(f"❌ OpenAI Error: {e}")

# Validar Search
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")

try:
    search_client = SearchClient(search_endpoint, "test-index", AzureKeyCredential(search_key))
    # Intentamos una operación básica
    print("✅ Azure Search: Conectado")
except Exception as e:
    print(f"❌ Search Error: {e}")