import numpy as np
from scipy.stats import norm
from config.settings import cfg
from src.models.paa import paa_transform


def get_breakpoints(alphabet_size: int) -> np.ndarray:
    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    return breakpoints


def series_to_sax(series: np.ndarray, n_segments: int = None, alphabet_size: int = None) -> str:
    if n_segments is None:
        n_segments = cfg["automata"]["paa_segments"]
    if alphabet_size is None:
        alphabet_size = cfg["automata"]["alphabet_size"]

    mean = np.mean(series)
    std = np.std(series)
    if std == 0:
        normalized = series - mean
    else:
        normalized = (series - mean) / std

    paa = paa_transform(normalized, n_segments)
    breakpoints = get_breakpoints(alphabet_size)

    letters = []
    for val in paa:
        idx = np.searchsorted(breakpoints, val)
        letters.append(chr(ord('a') + idx))

    return ''.join(letters)
