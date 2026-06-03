import numpy as np
import time
from src.models.automata import ProbabilisticAutomata
from src.models.pattern_extractor import extract_patterns, build_sax_vocabulary
from src.explainability.explainer import AutomataExplainer
from src.explainability.formatter import explain_sequence
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.logger import log_experiment
from config.settings import cfg


def run_automata(data: tuple, dataset: str, seed: int, scenario: str) -> dict:
    """
    Otomata modelini verilen veri üzerinde eğitip değerlendirir.
    data: (X_train, y_train, X_val, y_val, X_test, y_test)
    Not: X burada tek boyutlu (PC1) olmalıdır.
    """
    np.random.seed(seed)

    X_tr, y_tr, X_val, y_val, X_te, y_te = data
    window_size = cfg["automata"]["window_size"]

    # Train pattern'ları çıkar — sadece train'e fit
    t_start = time.time()
    train_series = X_tr.flatten()
    train_patterns = extract_patterns(train_series, window_size=window_size)

    # Sözlük ve otomata sadece train üzerinde oluşturulur
    vocabulary = build_sax_vocabulary(train_patterns)

    automata = ProbabilisticAutomata()
    automata.fit(train_patterns)
    automata.get_transition_probs()
    train_time = time.time() - t_start

    # Test pattern'ları çıkar ve tahmin yap
    t_start = time.time()
    test_series = X_te.flatten()
    test_patterns = extract_patterns(test_series, window_size=window_size)
    preds, probs = automata.predict(test_patterns)
    inference_time = time.time() - t_start

    # Etiketleri hizala
    y_true = y_te[window_size - 1: window_size - 1 + len(preds)]

    metrics = compute_all_metrics(y_true, preds)
    metrics["train_time"] = round(train_time, 4)
    metrics["inference_time"] = round(inference_time, 4)
    metrics["vocabulary_size"] = len(vocabulary)

    unseen_count = sum(1 for p in test_patterns if p not in vocabulary)
    total = len(test_patterns)
    metrics["detection_rate"] = round(unseen_count / total, 4) if total > 0 else 0.0

    mapped_correctly = sum(
        1 for p in test_patterns
        if p not in vocabulary and automata.resolve_pattern(p)[1] != "unknown"
    )
    metrics["mapping_accuracy"] = round(mapped_correctly / unseen_count, 4) if unseen_count > 0 else 1.0

    log_experiment("Automata", dataset, seed, scenario, metrics)

    # Açıklanabilirlik
    explainer = AutomataExplainer(automata)
    explanations = explain_sequence(explainer, test_patterns, window_size=window_size)

    return {
        "model": "Automata",
        "metrics": metrics,
        "preds": preds,
        "y_true": y_true,
        "probs": probs,
        "vocabulary": vocabulary,
        "explanations": explanations
    }
