# src/evaluator.py
import json
import logging

logger = logging.getLogger("fuyu-eval")

async def evaluate_rag(client, model, question, context, answer):
    # Definimos un esquema estricto para el LLM
    system_prompt = (
        "Eres un auditor experto en MLOps. Tu única tarea es evaluar la respuesta de un sistema RAG. "
        "Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, siguiendo este esquema: "
        "{'fidelidad': int, 'relevancia': int, 'razonamiento': string}. "
        "Las notas son del 1 al 10."
    )
    
    eval_prompt = f"Pregunta: {question}\nContexto: {context}\nRespuesta: {answer}"
    
    try:
        res = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": eval_prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        # Parseamos el JSON
        data = json.loads(res.choices[0].message.content)
        
        # Usamos .get() para evitar el KeyError si el LLM olvida una llave
        return {
            "fidelidad": data.get("fidelidad", 0),
            "relevancia": data.get("relevancia", 0),
            "razonamiento": data.get("razonamiento", "No se pudo extraer razonamiento.")
        }
        
    except Exception as e:
        logger.error(f"❌ Error crítico en evaluación: {e}")
        # Si todo falla, devolvemos un objeto seguro para que app.py no explote
        return {"fidelidad": 0, "relevancia": 0, "razonamiento": "Falla técnica en el auditor."}