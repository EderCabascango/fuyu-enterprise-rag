import os
import json
import logging
import chainlit as cl
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI 
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient 
#from azure.monitor.opentelemetry import configure_azure_monitor

# 1. FILTRADO DE LOGS (Crucial para MLOps)
# Esto silencia el ruido de "Transmission succeeded" que vimos en tu captura
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.WARNING)

load_dotenv()

# 2. ACTIVAR MONITOREO
#if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
#    configure_azure_monitor()
#    logging.info("🚀 MLOps: Telemetría conectada y activa.")

# Configuración de constantes para evitar errores de "hard-coding"
MODEL_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

aoai_client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
    api_version="2024-08-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# --- FUNCIÓN DE EVALUACIÓN ---
async def evaluate_rag(question, context, answer):
    eval_prompt = f"""
    Actúa como un auditor de calidad de IA. Evalúa la respuesta basándote SOLO en el contexto.
    Pregunta: {question}
    Contexto: {context}
    Respuesta: {answer}
    Devuelve un JSON con: fidelidad (0-10), relevancia (0-10) y razonamiento.
    """
    
    try:
        res = await aoai_client.chat.completions.create(
            model=MODEL_NAME, # Usamos la constante
            messages=[{"role": "system", "content": "Eres un experto en MLOps. Responde solo en JSON."},
                      {"role": "user", "content": eval_prompt}],
            response_format={ "type": "json_object" }
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        logging.error(f"Error en LLM-as-a-Judge: {e}")
        return {"fidelidad": 0, "relevancia": 0, "razonamiento": "Error en evaluación."}

@cl.on_chat_start
async def start():
    logging.info("Nueva sesión iniciada.")
    await cl.Message(content="🤖 **Tutor AI-300 + Auditor MLOps**. Sistema listo y monitoreado.").send()

@cl.on_message
async def main(message: cl.Message):
    print(f">>> [CONTROL] Mensaje recibido: {message.content}") 
    
    # 1. Prueba de Vectorización
    print(">>> [CONTROL] Intentando vectorizar...")
    try:
        emb = await aoai_client.embeddings.create(
            input=message.content, 
            model="text-embedding-3-large" 
        )
        print(">>> [CONTROL] Vectorización ✅")
    except Exception as e:
        print(f">>> [CONTROL] ERROR en Embeddings ❌: {e}")
        await cl.Message(content=f"❌ Error en Embeddings: {e}").send()
        return

    # 2. Prueba de Búsqueda
    print(">>> [CONTROL] Intentando buscar en AI Search...")
    context_list = []
    try:
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name="fuyu-enterprise-index",
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
        )
        async with search_client:
            results = await search_client.search(
                search_text=message.content,
                vector_queries=[{"vector": emb.data[0].embedding, "fields": "content_vector", "k": 3, "kind": "vector"}],
                top=3,
                select=["content"]
            )
            async for res in results:
                context_list.append(res["content"])
        print(f">>> [CONTROL] Search ✅ ({len(context_list)} docs)")
    except Exception as e:
        print(f">>> [CONTROL] ERROR en Search ❌: {e}")
        await cl.Message(content=f"❌ Error en Search: {e}").send()
        return

    context = "\n\n".join(context_list) if context_list else "Sin contexto."

    # 3. Respuesta Final
    print(">>> [CONTROL] Generando respuesta con GPT...")
    try:
        response = await aoai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "Responde solo basado en el contexto."},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {message.content}"}
            ]
        )
        await cl.Message(content=response.choices[0].message.content).send()
        print(">>> [CONTROL] Proceso completado ✅")
    except Exception as e:
        print(f">>> [CONTROL] ERROR en Chat ❌: {e}")
        await cl.Message(content=f"❌ Error en Chat: {e}").send()