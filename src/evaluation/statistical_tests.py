import numpy as np
from scipy.stats import wilcoxon, chi2


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


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray, alpha: float = 0.05) -> dict:
    """
    İki modelin tahminleri arasında McNemar testi uygular.
    y_pred_a, y_pred_b: iki farklı modelin tahminleri
    """
    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Uyuşmazlık tablosu
    b = np.sum(correct_a & ~correct_b)   # A doğru, B yanlış
    c = np.sum(~correct_a & correct_b)   # A yanlış, B doğru

    if b + c == 0:
        return {
            "test": "mcnemar",
            "b": int(b), "c": int(c),
            "statistic": 0.0,
            "p_value": 1.0,
            "alpha": alpha,
            "significant": False,
            "conclusion": "Anlamlı fark yok"
        }

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {
        "test": "mcnemar",
        "b": int(b), "c": int(c),
        "statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "significant": p_value < alpha,
        "conclusion": "Anlamlı fark var" if p_value < alpha else "Anlamlı fark yok"
    }
