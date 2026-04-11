import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

# Configuración de Clientes
aoai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-08-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name="fuyu-enterprise-index",
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

def ask_ai_300(question):
    # 1. Generar embedding de la pregunta
    query_vector = aoai_client.embeddings.create(
        input=question, model="text-embedding-3-large"
    ).data[0].embedding

    # 2. Buscar en Azure AI Search (Top 3 fragmentos mas relevantes)
    results = search_client.search(
        search_text=None,
        vector_queries=[{
            "vector": query_vector,
            "fields": "content_vector",
            "k": 3,
            "kind": "vector"
        }],
        select=["content"]
    )
    
    context = "\n\n".join([res["content"] for res in results])

    # 3. Generar respuesta con GPT-4o-mini
    response = aoai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un tutor experto en la certificación AI-300 de Azure. Responde de forma técnica y precisa basándote solo en el contexto proporcionado."},
            {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"}
        ]
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    print("🤖 Bot AI-300 Listo. (Escribe 'salir' para terminar)")
    while True:
        user_input = input("\nTu pregunta: ")
        if user_input.lower() == "salir": break
        print("\n🔍 Pensando...")
        respuesta = ask_ai_300(user_input)
        print(f"\n🎓 Respuesta:\n{respuesta}")