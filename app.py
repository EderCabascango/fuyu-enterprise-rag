import os
import json
import logging
import chainlit as cl
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI 
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient 
from azure.monitor.opentelemetry import configure_azure_monitor

load_dotenv()

# 1. ACTIVAR MONITOREO (Debe ir antes de cualquier cliente)
# Esto enviará logs, métricas y trazas automáticamente a Application Insights
if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor()
    logging.info("MLOps: Observabilidad de Azure activada.")

# 2. CONFIGURACIÓN DE CLIENTES (Mapeo de variables de Azure)
# Ajusté "AZURE_OPENAI_KEY" a "AZURE_OPENAI_API_KEY" para que coincida con tu terminal
aoai_client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
    api_version="2024-08-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# --- FUNCIÓN DE EVALUACIÓN (LLM-AS-A-JUDGE) ---
async def evaluate_rag(question, context, answer):
    eval_prompt = f"""
    Actúa como un auditor de calidad de IA. Evalúa la siguiente respuesta basándote SOLO en el contexto proporcionado.
    
    Pregunta: {question}
    Contexto: {context}
    Respuesta: {answer}
    
    Devuelve un JSON con este formato:
    {{
        "fidelidad": (puntuación 0-10 si la respuesta está en el contexto),
        "relevancia": (puntuación 0-10 si responde a la duda),
        "razonamiento": (breve explicación de por qué esa nota)
    }}
    """
    
    res = await aoai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
        messages=[{"role": "system", "content": "Eres un experto en MLOps evaluando RAG. Responde solo en JSON."},
                  {"role": "user", "content": eval_prompt}],
        response_format={ "type": "json_object" }
    )
    return json.loads(res.choices[0].message.content)

@cl.on_chat_start
async def start():
    logging.info("Sesión de chat iniciada por el usuario.")
    await cl.Message(content="🤖 **Tutor AI-300 + Auditor MLOps**. Ahora evaluaré cada respuesta en tiempo real.").send()

@cl.on_message
async def main(message: cl.Message):
    # Inicializamos el cliente de búsqueda dentro del mensaje
    search_client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name="fuyu-enterprise-index",
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
    )

    # 1. Vectorizar (Usamos el modelo text-embedding-3-large como definiste)
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
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
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

    # 4. EVALUACIÓN AUTOMÁTICA
    try:
        evaluation = await evaluate_rag(message.content, context, full_answer)
        
        eval_text = f"""
### 📊 Métricas de Calidad
* **Fidelidad:** {evaluation['fidelidad']}/10
* **Relevancia:** {evaluation['relevancia']}/10

**Razonamiento:**
{evaluation['razonamiento']}
        """
        
        source_elements = [cl.Text(name="⚖️ Evaluación MLOps", content=eval_text, display="side")]
        for i, text in enumerate(context_list):
            source_elements.append(cl.Text(name=f"Fuente {i+1}", content=text, display="side"))

        msg.elements = source_elements
        await msg.update()
        logging.info(f"Respuesta generada y evaluada. Fidelidad: {evaluation['fidelidad']}")

    except Exception as e:
        logging.error(f"Error en la evaluación: {e}")
        await cl.ErrorMessage(content="Error al generar métricas de auditoría.").send()