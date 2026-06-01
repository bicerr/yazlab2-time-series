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
