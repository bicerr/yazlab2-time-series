import numpy as np
from config.settings import cfg


def paa_transform(series: np.ndarray, n_segments: int = None) -> np.ndarray:
    if n_segments is None:
        n_segments = cfg["automata"]["paa_segments"]

    length = len(series)
    if length < n_segments:
        raise ValueError(f"Seri uzunluğu ({length}) segment sayısından ({n_segments}) küçük olamaz.")

    segment_size = length / n_segments
    paa = np.zeros(n_segments)

    for i in range(n_segments):
        start = int(i * segment_size)
        end = int((i + 1) * segment_size)
        paa[i] = np.mean(series[start:end])

    return paa
