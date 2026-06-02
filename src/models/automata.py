import numpy as np
from collections import defaultdict
from src.models.levenshtein import find_nearest_pattern


class ProbabilisticAutomata:
    def __init__(self):
        self.states = set()
        self.transition_counts = defaultdict(lambda: defaultdict(int))

    def fit(self, patterns: list):
        self.states = set(patterns)

        for i in range(len(patterns) - 1):
            current = patterns[i]
            next_state = patterns[i + 1]
            self.transition_counts[current][next_state] += 1

    def get_transition_probs(self, laplace_alpha: float = 1.0) -> dict:
        probs = {}
        all_states = list(self.states)
        n_states = len(all_states)

        for state in all_states:
            counts = self.transition_counts[state]
            total = sum(counts.values()) + laplace_alpha * n_states
            probs[state] = {}
            for next_state in all_states:
                probs[state][next_state] = (counts.get(next_state, 0) + laplace_alpha) / total

        self.transition_probs = probs
        return probs

    def compute_path_probability(self, patterns: list) -> float:
        if not hasattr(self, 'transition_probs'):
            self.get_transition_probs()

        prob = 1.0
        for i in range(len(patterns) - 1):
            current = patterns[i]
            next_state = patterns[i + 1]
            if current in self.transition_probs and next_state in self.transition_probs[current]:
                prob *= self.transition_probs[current][next_state]
            else:
                prob *= 1e-10
        return prob

    def resolve_pattern(self, pattern: str) -> tuple:
        """
        Pattern sözlükte varsa döndürür.
        Yoksa Levenshtein ile en yakın pattern'a map eder.
        Returns: (resolved_pattern, status, mapped_to, distance)
        """
        if pattern in self.states:
            return pattern, "known", None, 0
        else:
            nearest, dist = find_nearest_pattern(pattern, self.states)
            return nearest, "unseen", nearest, dist

    def predict(self, patterns: list, threshold: float = None) -> tuple:
        from config.settings import cfg
        if threshold is None:
            threshold = cfg["automata"].get("anomaly_threshold", 1e-5)

        if not hasattr(self, 'transition_probs'):
            self.get_transition_probs()

        predictions = []
        probabilities = []

        window = cfg["automata"]["window_size"]
        for i in range(len(patterns) - window + 1):
            seq = patterns[i:i + window]
            # Unseen pattern'ları en yakın bilinen pattern'a map et
            resolved_seq = [self.resolve_pattern(p)[0] for p in seq]
            prob = self.compute_path_probability(resolved_seq)
            probabilities.append(prob)
            predictions.append(1 if prob < threshold else 0)

        return np.array(predictions), np.array(probabilities)

    def resolve_pattern(self, pattern: str, max_distance: int = 3) -> tuple:
        from src.models.levenshtein import find_nearest_pattern

        if pattern in self.states:
            return pattern, "known", pattern, 0

        nearest, dist = find_nearest_pattern(pattern, self.states)
        if dist <= max_distance:
            return nearest, "unseen", nearest, dist

        return pattern, "unknown", None, dist
