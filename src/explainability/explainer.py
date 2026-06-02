from src.models.automata import ProbabilisticAutomata
from src.models.levenshtein import find_nearest_pattern
from config.settings import cfg


class AutomataExplainer:
    def __init__(self, automata: ProbabilisticAutomata):
        self.automata = automata
        if not hasattr(automata, 'transition_probs'):
            automata.get_transition_probs()

    def explain_step(self, patterns: list, time_step: int) -> dict:
        """
        Tek bir zaman adımı için açıklama üretir.
        patterns: o anki sliding window'daki pattern listesi
        """
        threshold = cfg["automata"].get("anomaly_threshold", 1e-5)

        transitions = []
        resolved_patterns = []

        for p in patterns:
            resolved, status, mapped_to, dist = self.automata.resolve_pattern(p)
            resolved_patterns.append({
                "original": p,
                "resolved": resolved,
                "status": status,
                "mapped_to": mapped_to,
                "distance": dist
            })

        # Geçişleri ve olasılıklarını hesapla
        for i in range(len(resolved_patterns) - 1):
            src = resolved_patterns[i]["resolved"]
            dst = resolved_patterns[i + 1]["resolved"]
            prob = self.automata.transition_probs.get(src, {}).get(dst, 1e-10)
            transitions.append({
                "from": src,
                "to": dst,
                "probability": prob
            })

        # Path probability
        path_prob = self._compute_path_prob(transitions)

        # Confidence score ve karar
        decision = "anomaly" if path_prob < threshold else "normal"
        confidence = path_prob

        return {
            "time_step": time_step,
            "state": resolved_patterns[0]["resolved"] if resolved_patterns else None,
            "pattern": patterns[0] if patterns else None,
            "status": resolved_patterns[0]["status"] if resolved_patterns else None,
            "mapped_to": resolved_patterns[0]["mapped_to"] if resolved_patterns else None,
            "transitions": transitions,
            "path_probability": path_prob,
            "confidence_score": confidence,
            "decision": decision
        }

    def _compute_path_prob(self, transitions: list) -> float:
        prob = 1.0
        for t in transitions:
            prob *= t["probability"]
        return prob
