import os
import json
import logging
import chainlit as cl
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI 
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient 
from azure.monitor.opentelemetry import configure_azure_monitor

# 1. FILTRADO DE LOGS (Crucial para MLOps)
# Esto silencia el ruido de "Transmission succeeded" que vimos en tu captura
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.WARNING)

load_dotenv()

# 2. ACTIVAR MONITOREO
if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor()
    logging.info("🚀 MLOps: Telemetría conectada y activa.")

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
    search_client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name="fuyu-enterprise-index",
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
    )

    # 1. Vectorizar
    emb = await aoai_client.embeddings.create(input=message.content, model="text-embedding-3-large")
    
    # 2. Búsqueda Híbrida
    context_list = []
    async with search_client:
        results = await search_client.search(
            search_text=message.content,
            vector_queries=[{"vector": emb.data[0].embedding, "fields": "content_vector", "k": 3, "kind": "vector"}],
            top=3,
            select=["content"]
        )
        async for res in results:
            context_list.append(res["content"])
    
    context = "\n\n".join(context_list) if context_list else "Sin contexto."

    # 3. Respuesta en Streaming
    msg = cl.Message(content="")
    full_answer = ""
    
    response = await aoai_client.chat.completions.create(
        model=MODEL_NAME, # Usamos la constante
        messages=[
            {"role": "system", "content": "Responde solo basado en el contexto. Cita fuentes como (Fuente 1)."},
            {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {message.content}"}
        ],
        stream=True
    )

    async for part in response:
        if part.choices and part.choices[0].delta.content:
            token = part.choices[0].delta.content
            full_answer += token
            await msg.stream_token(token)

    # 4. EVALUACIÓN Y CIERRE DE LOGS
    evaluation = await evaluate_rag(message.content, context, full_answer)
    
    eval_text = f"### 📊 Métricas\n* **Fidelidad:** {evaluation['fidelidad']}\n* **Relevancia:** {evaluation['relevancia']}\n\n**Razonamiento:**\n{evaluation['razonamiento']}"
    
    source_elements = [cl.Text(name="⚖️ Evaluación", content=eval_text, display="side")]
    for i, text in enumerate(context_list):
        source_elements.append(cl.Text(name=f"Fuente {i+1}", content=text, display="side"))

    msg.elements = source_elements
    await msg.update()
    
    # Este log ahora viajará limpio a App Insights
    logging.info(f"MLOps Eval - Q: {message.content[:30]}... | F: {evaluation['fidelidad']}")