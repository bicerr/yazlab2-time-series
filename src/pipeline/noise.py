import numpy as np
from config.settings import cfg


def add_gaussian_noise(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Veriye Gaussian gürültü ekler.
    std: config'den alınan noise_std değeri
    """
    np.random.seed(seed)
    std = cfg["experiment"]["noise_std"]
    noise = np.random.normal(0, std, X.shape)
    return X + noise
