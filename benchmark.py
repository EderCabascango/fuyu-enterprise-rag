import asyncio
import json
import pandas as pd
from src.engine import RAGEngine
from src.evaluator import evaluate_rag

async def run_benchmark():
    engine = RAGEngine()
    with open("tests/golden_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    results = []
    print(f"🚀 Iniciando Benchmark de {len(dataset)} preguntas...")

    for item in dataset:
        print(f"🧐 Procesando: {item['question']}")
        
        # 1. Obtener contexto y respuesta
        context_list = await engine.get_context(item['question'])
        context = "\n\n".join(context_list)
        
        # Generamos respuesta (llamada directa al cliente de OpenAI del engine)
        res = await engine.client.chat.completions.create(
            model=engine.model,
            messages=[
                {"role": "system", "content": "Responde solo con el contexto proporcionado."},
                {"role": "user", "content": f"Contexto: {context}\nPregunta: {item['question']}"}
            ]
        )
        answer = res.choices[0].message.content

        # 2. Evaluación automática
        eval_result = await evaluate_rag(engine.client, engine.model, item['question'], context, answer)

        # 3. Guardar datos
        results.append({
            "Pregunta": item['question'],
            "Respuesta_Bot": answer,
            "Fidelidad": eval_result['fidelidad'],
            "Relevancia": eval_result['relevancia'],
            "Razonamiento": eval_result['razonamiento']
        })

    # 4. Exportar a CSV para tu análisis de Data Science
    df = pd.DataFrame(results)
    df.to_csv("tests/benchmark_results.csv", index=False)
    print("✅ Benchmark completado. Resultados guardados en tests/benchmark_results.csv")
    print(f"📊 Promedio Fidelidad: {df['Fidelidad'].mean():.2f}/10")

if __name__ == "__main__":
    asyncio.run(run_benchmark())