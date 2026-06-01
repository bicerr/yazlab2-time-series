import numpy as np
from collections import defaultdict


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
