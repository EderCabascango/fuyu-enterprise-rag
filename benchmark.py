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

    # 4. Exportar a CSV para análisis
    df = pd.DataFrame(results)
    df.to_csv("tests/benchmark_results.csv", index=False)
    
    # 5. Generar Reporte Profesional en Markdown
    avg_fidelidad = df['Fidelidad'].mean()
    avg_relevancia = df['Relevancia'].mean()
    
    report_md = f"""# 📊 Reporte de Benchmark - Fuyu RAG
Generado automáticamente el: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📈 Resumen de Calidad
| Métrica | Promedio | Meta | Estatus |
|---------|----------|------|---------|
| **Fidelidad** | {avg_fidelidad:.2f}/10 | 8.5 | {'✅' if avg_fidelidad >= 8.5 else '⚠️'} |
| **Relevancia** | {avg_relevancia:.2f}/10 | 9.0 | {'✅' if avg_relevancia >= 9.0 else '⚠️'} |

## 📝 Detalle por Pregunta
"""
    for index, row in df.iterrows():
        report_md += f"### {index + 1}. {row['Pregunta']}\n"
        report_md += f"- **Audit Scores:** Fidelidad: {row['Fidelidad']} | Relevancia: {row['Relevancia']}\n"
        report_md += f"- **Razonamiento:** {row['Razonamiento']}\n\n"

    with open("BENCHMARK_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("✅ Benchmark completado.")
    print("📊 Resultados guardados en tests/benchmark_results.csv")
    print("📋 Reporte profesional generado en BENCHMARK_REPORT.md")
    print(f"⭐ Promedio Fidelidad: {avg_fidelidad:.2f}/10")

if __name__ == "__main__":
    asyncio.run(run_benchmark())