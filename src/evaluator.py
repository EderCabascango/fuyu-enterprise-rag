# src/evaluator.py
import json
import logging

logger = logging.getLogger("fuyu-eval")

async def evaluate_response(client, model, question, context, answer):
    eval_prompt = f"Evaluación de RAG... Pregunta: {question} | Contexto: {context} | Respuesta: {answer}"
    try:
        res = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "Eres un auditor experto en MLOps. Responde solo en JSON."},
                      {"role": "user", "content": eval_prompt}],
            response_format={ "type": "json_object" }
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        logger.error(f"❌ Error en evaluación: {e}")
        return {"fidelidad": 0, "relevancia": 0, "razonamiento": "Error técnico."}