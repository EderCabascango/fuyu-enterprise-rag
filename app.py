import os
import json
import chainlit as cl
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI 
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient 

load_dotenv()

aoai_client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
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
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Eres un experto en MLOps evaluando RAG. Responde solo en JSON."},
                  {"role": "user", "content": eval_prompt}],
        response_format={ "type": "json_object" }
    )
    return json.loads(res.choices[0].message.content)

@cl.on_chat_start
async def start():
    await cl.Message(content="🤖 **Tutor AI-300 + Auditor MLOps**. Ahora evaluaré cada respuesta en tiempo real.").send()

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
        model="gpt-4o-mini",
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
    evaluation = await evaluate_rag(message.content, context, full_answer)
    
    # Preparamos el panel lateral con Fuentes y Métricas
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