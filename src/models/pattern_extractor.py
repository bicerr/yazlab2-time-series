import numpy as np
from config.settings import cfg
from src.models.sax import series_to_sax


def extract_patterns(X: np.ndarray, window_size: int = None, n_segments: int = None, alphabet_size: int = None) -> list:
    if window_size is None:
        window_size = cfg["automata"]["window_size"]
    if n_segments is None:
        n_segments = cfg["automata"]["paa_segments"]
    if alphabet_size is None:
        alphabet_size = cfg["automata"]["alphabet_size"]

    patterns = []
    for i in range(len(X) - window_size + 1):
        window = X[i:i + window_size].flatten()
        pattern = series_to_sax(window, n_segments, alphabet_size)
        patterns.append(pattern)

    return patterns


def build_sax_vocabulary(patterns: list) -> set:
    return set(patterns)
