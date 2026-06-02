import json


def to_json(explanation: dict, indent: int = 2) -> str:
    """Açıklama sözlüğünü PDF'deki zorunlu JSON formatına dönüştürür."""
    output = {
        "time_step": explanation["time_step"],
        "state": explanation["state"],
        "pattern": explanation["pattern"],
        "status": explanation["status"],
        "mapped_to": explanation["mapped_to"],
        "transitions": [
            {
                "from": t["from"],
                "to": t["to"],
                "probability": round(t["probability"], 6)
            }
            for t in explanation["transitions"]
        ],
        "path_probability": round(explanation["path_probability"], 6),
        "confidence_score": round(explanation["confidence_score"], 6),
        "decision": explanation["decision"]
    }
    return json.dumps(output, indent=indent, ensure_ascii=False)


def to_table_row(explanation: dict) -> str:
    """Açıklamayı tek satır tablo formatında döndürür."""
    return (
        f"t={explanation['time_step']:>4} | "
        f"state={explanation['state']:>6} | "
        f"pattern={explanation['pattern']:>6} | "
        f"status={explanation['status']:>6} | "
        f"path_prob={explanation['path_probability']:.4e} | "
        f"decision={explanation['decision']}"
    )


def explain_sequence(explainer, patterns: list, window_size: int) -> list:
    """
    Tüm zaman serisi üzerinde açıklama üretir.
    Her window için bir açıklama döner.
    """
    explanations = []
    for i in range(len(patterns) - window_size + 1):
        window = patterns[i:i + window_size]
        explanation = explainer.explain_step(window, time_step=i)
        explanations.append(explanation)
    return explanations
