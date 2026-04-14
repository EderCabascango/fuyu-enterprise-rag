import os
import chainlit as cl
from src.engine import RAGEngine
from src.evaluator import evaluate_rag
from azure.monitor.opentelemetry import configure_azure_monitor

# Iniciamos monitoreo y motor
if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor()

engine = RAGEngine()

@cl.on_message
async def main(message: cl.Message):
    # 1. Recuperar Contexto
    context_list = await engine.get_context(message.content)
    context = "\n\n".join(context_list)

    # 2. Generar Respuesta con Streaming
    msg = cl.Message(content="")
    full_answer = ""
    
    response = await engine.client.chat.completions.create(
        model=engine.model,
        messages=[
            {"role": "system", "content": "Responde solo basado en el contexto. Cita fuentes (Fuente 1)."},
            {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {message.content}"}
        ],
        stream=True
    )

    async for part in response:
        if part.choices and part.choices[0].delta.content:
            token = part.choices[0].delta.content
            full_answer += token
            await msg.stream_token(token)

    # 3. Evaluación Automática (Auditoría)
    evaluation = await evaluate_rag(engine.client, engine.model, message.content, context, full_answer)
    
    # 4. Mostrar Resultados y Fuentes
    eval_text = f"### ⚖️ Auditoría\n* **Fidelidad:** {evaluation['fidelidad']}/10\n* **Relevancia:** {evaluation['relevancia']}/10"
    
    source_elements = [cl.Text(name="📊 Métricas", content=eval_text, display="side")]
    for i, text in enumerate(context_list):
        source_elements.append(cl.Text(name=f"Fuente {i+1}", content=text, display="side"))

    msg.elements = source_elements
    await msg.update()