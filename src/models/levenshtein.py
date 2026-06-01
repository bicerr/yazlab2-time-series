import numpy as np


def levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return int(dp[m][n])


def find_nearest_pattern(unseen: str, vocabulary: set) -> tuple:
    """
    Görülmemiş pattern'a en yakın bilinen pattern'ı döndürür.
    Returns: (nearest_pattern, distance)
    """
    if not vocabulary:
        raise ValueError("Sözlük boş olamaz.")

    best_pattern = None
    best_distance = float("inf")

    for known in vocabulary:
        dist = levenshtein_distance(unseen, known)
        if dist < best_distance:
            best_distance = dist
            best_pattern = known

    return best_pattern, best_distance
