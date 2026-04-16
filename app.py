import os
import chainlit as cl
from src.engine import RAGEngine
from src.evaluator import evaluate_rag
from src.utils import log_feedback, log_failure
from azure.monitor.opentelemetry import configure_azure_monitor

# 1. Iniciamos monitoreo y motor
if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor()

engine = RAGEngine()

@cl.on_chat_start
async def start():
    # En un entorno real, recuperarías el ID del usuario autenticado
    # Aquí usaremos un ID persistente de prueba para demostrar la funcionalidad
    user_id = "user-fuyu-001" 
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("engine", engine)
    
    # 1. Recuperar historial persistente si existe
    history = await engine.get_chat_history(user_id)
    cl.user_session.set("history", history)
    
    msg_content = "🤖 **Fuyu Tech Assistant** activo (Modo Persistente)."
    if history:
        msg_content += f"\nHe recuperado nuestra conversación anterior ({len(history)} mensajes). ¿En qué nos quedamos?"
    else:
        msg_content += "\nNo encontré conversaciones previas. ¿En qué duda técnica puedo ayudarte hoy?"
        
    await cl.Message(content=msg_content).send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    user_id = cl.user_session.get("user_id")
    
    # 1. Recuperar Contexto con Re-ranking Semántico
    context_list = await engine.get_context(message.content)
    context = "\n\n".join(context_list)

    # 2. Instrucción del Sistema
    system_instruction = (
        "Eres un tutor experto en Azure y MLOps. Responde basándote ÚNICAMENTE en el contexto.\n"
        "Si no está en el contexto, di que no tienes información fáctica.\n"
        "Cita tus fuentes como (Fuente X)."
    )
    
    messages = [{"role": "system", "content": system_instruction}]
    for h in history[-5:]: # Enviamos más contexto histórico
        messages.append({"role": h["role"], "content": h["content"]})
    
    messages.append({"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {message.content}"})

    # 3. Generar Respuesta
    msg = cl.Message(content="")
    full_answer = ""
    
    response = await engine.client.chat.completions.create(
        model=engine.model,
        messages=messages,
        stream=True
    )

    async for part in response:
        if part.choices and part.choices[0].delta.content:
            token = part.choices[0].delta.content
            full_answer += token
            await msg.stream_token(token)

    # 4. Evaluación Automática
    evaluation = await evaluate_rag(engine.client, engine.model, message.content, context, full_answer)
    
    # 5. Guardar en Sesión y PERSISTIR en Cosmos DB
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": full_answer})
    cl.user_session.set("history", history)
    
    await engine.save_chat_history(user_id, history)

    # 6. Guardar metadatos para Feedback
    cl.user_session.set("last_query", message.content)
    cl.user_session.set("last_answer", full_answer)
    cl.user_session.set("last_evaluation", evaluation)
    cl.user_session.set("last_context", context)

    # 7. Mostrar Resultados
    eval_text = (
        f"### ⚖️ Auditoría de IA\n"
        f"* **Fidelidad:** {evaluation['fidelidad']}/10\n"
        f"* **Relevancia:** {evaluation['relevancia']}/10\n\n"
        f"**Nota del Auditor:** {evaluation['razonamiento']}"
    )
    
    source_elements = [cl.Text(name="📊 Métricas Calidad", content=eval_text, display="side")]
    for i, text in enumerate(context_list):
        source_elements.append(cl.Text(name=f"Fuente {i+1}", content=text, display="side"))

    msg.elements = source_elements
    await msg.update()

@cl.on_feedback
async def feedback(feedback):
    # Recuperar datos de la sesión para el log
    query = cl.user_session.get("last_query")
    answer = cl.user_session.get("last_answer")
    evaluation = cl.user_session.get("last_evaluation")
    context = cl.user_session.get("last_context")

    # Guardar feedback
    log_feedback(feedback, feedback.forId, query, answer, evaluation)
    
    # Si es negativo, lo tratamos como reporte de falla
    if feedback.value == 0:
        log_failure(feedback.forId, query, answer, context, feedback.comment)
        await cl.Message(content="⚠️ Gracias por reportar la falla. El equipo de MLOps revisará este caso para mejorar el modelo.").send()
    else:
        await cl.Message(content="✅ ¡Gracias por tu feedback positivo!").send()

@cl.on_chat_end
async def end():
    await engine.close()