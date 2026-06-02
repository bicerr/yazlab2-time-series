import numpy as np
from scipy.stats import wilcoxon


def wilcoxon_test(scores_a: list, scores_b: list, alpha: float = 0.05) -> dict:
    """
    İki modelin F1 skorları arasında Wilcoxon işaretli sıra testi uygular.
    scores_a, scores_b: 5 seed ile elde edilen F1 skorları listesi
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("İki skor listesi aynı uzunlukta olmalıdır.")

    stat, p_value = wilcoxon(scores_a, scores_b)

    return {
        "test": "wilcoxon",
        "statistic": stat,
        "p_value": p_value,
        "alpha": alpha,
        "significant": p_value < alpha,
        "conclusion": "Anlamlı fark var" if p_value < alpha else "Anlamlı fark yok"
    }
