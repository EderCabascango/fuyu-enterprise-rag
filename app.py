import os
import chainlit as cl
from src.engine import RAGEngine
from src.evaluator import evaluate_rag
from azure.monitor.opentelemetry import configure_azure_monitor

# 1. Iniciamos monitoreo y motor
# Solo configuramos Azure Monitor si existe la cadena de conexión
if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor()

# Instanciamos el motor globalmente para reusar conexiones
engine = RAGEngine()

@cl.on_chat_start
async def start():
    cl.user_session.set("engine", engine)
    await cl.Message(content="🤖 Tutor de MLOps y Azure activo. ¿En qué puedo ayudarte hoy?").send()

@cl.on_message
async def main(message: cl.Message):
    # 1. Recuperar Contexto desde Azure AI Search
    context_list = await engine.get_context(message.content)
    context = "\n\n".join(context_list)

    # 2. Instrucción del Sistema (Alineada para subir nota de Benchmark)
    system_instruction = (
        "Eres un tutor experto en Azure y MLOps. Tu objetivo es responder dudas técnicas "
        "con precisión quirúrgica basándote ÚNICAMENTE en el contexto proporcionado.\n\n"
        "REGLAS DE ORO:\n"
        "1. Si la información NO está en el contexto, di: 'Lo siento, no cuento con información fáctica en mis fuentes para responder esto'.\n"
        "2. PROHIBIDO: Inferir, suponer o usar conocimiento general para llenar vacíos. Si el contexto no define un término, no lo definas tú.\n"
        "3. PRECISIÓN TÉCNICA: Usa los nombres exactos de roles y servicios que aparezcan en el contexto.\n"
        "4. CITAS: Al final de cada párrafo fáctico, añade (Fuente X).\n"
        "5. Sé directo y evita preámbulos innecesarios."
    )

    # 3. Generar Respuesta con Streaming
    msg = cl.Message(content="")
    full_answer = ""
    
    response = await engine.client.chat.completions.create(
        model=engine.model,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {message.content}"}
        ],
        stream=True
    )

    async for part in response:
        if part.choices and part.choices[0].delta.content:
            token = part.choices[0].delta.content
            full_answer += token
            await msg.stream_token(token)

    # 4. Evaluación Automática (Auditoría de Calidad)
    # Usamos la versión robusta que previene KeyErrors
    evaluation = await evaluate_rag(engine.client, engine.model, message.content, context, full_answer)
    
    f = evaluation.get('fidelidad', 0)
    r = evaluation.get('relevancia', 0)
    razon = evaluation.get('razonamiento', 'Sin detalles del auditor.')

    # 5. Mostrar Resultados y Fuentes en el Panel Lateral
    eval_text = (
        f"### ⚖️ Auditoría de IA\n"
        f"* **Fidelidad:** {f}/10\n"
        f"* **Relevancia:** {r}/10\n\n"
        f"**Nota del Auditor:** {razon}"
    )
    
    source_elements = [cl.Text(name="📊 Métricas Calidad", content=eval_text, display="side")]
    
    # Añadimos los fragmentos de documentos encontrados
    for i, text in enumerate(context_list):
        source_elements.append(cl.Text(name=f"Fuente {i+1}", content=text, display="side"))

    msg.elements = source_elements
    await msg.update()

@cl.on_chat_end
async def end():
    # Limpieza de sesiones al cerrar el chat
    await engine.close()

# Versión forzada 4.0 - RBAC activo y Prompt Optimizado