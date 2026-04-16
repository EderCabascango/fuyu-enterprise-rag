import csv
import os
import json
from datetime import datetime

FEEDBACK_FILE = "data/feedback_logs.csv"

def log_feedback(user_feedback, message_id, question, answer, evaluation):
    """
    Guarda el feedback del usuario en un archivo CSV para posterior análisis.
    """
    file_exists = os.path.isfile(FEEDBACK_FILE)
    
    # Asegurar que el directorio existe
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    
    with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "message_id", "value", "comment", "question", "answer", "fidelidad", "relevancia"])
            
        writer.writerow([
            datetime.now().isoformat(),
            message_id,
            user_feedback.get("value"),
            user_feedback.get("comment", ""),
            question,
            answer,
            evaluation.get("fidelidad"),
            evaluation.get("relevancia")
        ])

def log_failure(message_id, question, answer, context, reason):
    """
    Logs a technical failure or hallucination report.
    """
    failure_log = "data/failure_reports.jsonl"
    os.makedirs(os.path.dirname(failure_log), exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "message_id": message_id,
        "question": question,
        "answer": answer,
        "context": context,
        "reason": reason
    }
    
    with open(failure_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")
