import numpy as np
from src.pipeline.pipeline import prepare_batadal, prepare_skab
from src.pipeline.automata_runner import run_automata
from src.preprocessing.preprocessor import fit_transform_scaler, fit_transform_pca
from src.models.automata import ProbabilisticAutomata
from src.models.pattern_extractor import extract_patterns, build_sax_vocabulary
from src.evaluation.metrics import compute_all_metrics
from config.settings import cfg


def run_sweep_single(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     window_size: int, alphabet_size: int, seed: int) -> dict:
    """
    Belirli window_size ve alphabet_size ile otomata modelini çalıştırır.
    """
    np.random.seed(seed)

    train_series = X_train.flatten()
    test_series = X_test.flatten()

    # n_segments = window_size olarak ayarla (pencere boyutuna uyum)
    train_patterns = extract_patterns(
        train_series, window_size=window_size,
        n_segments=window_size, alphabet_size=alphabet_size
    )
    test_patterns = extract_patterns(
        test_series, window_size=window_size,
        n_segments=window_size, alphabet_size=alphabet_size
    )

    vocabulary = build_sax_vocabulary(train_patterns)

    automata = ProbabilisticAutomata()
    automata.fit(train_patterns)
    automata.get_transition_probs()

    preds, probs = automata.predict(test_patterns)
    y_true = y_test[window_size - 1: window_size - 1 + len(preds)]

    metrics = compute_all_metrics(y_true, preds)
    metrics["state_count"] = len(automata.states)
    metrics["vocabulary_size"] = len(vocabulary)

    total_transitions = sum(
        sum(targets.values())
        for targets in automata.transition_counts.values()
    )
    max_transitions = len(automata.states) ** 2
    metrics["transition_density"] = round(total_transitions / max_transitions, 4) if max_transitions > 0 else 0.0

    return metrics


def run_parameter_sweep(dataset: str = "BATADAL", seed: int = 42) -> dict:
    """
    window_size ve alphabet_size kombinasyonlarını tarar.
    PDF: window ∈ {3,4,5,6}, alphabet ∈ {3,4,5,6}
    """
    window_sizes = cfg["parameter_sweep"]["window_sizes"]
    alphabet_sizes = cfg["parameter_sweep"]["alphabet_sizes"]

    if dataset == "BATADAL":
        data = prepare_batadal()
        X_tr, y_tr, _, _, X_te, y_te = data["automata"]
        runs = [(X_tr, y_tr, X_te, y_te)]

    elif dataset == "SKAB":
        folds = prepare_skab()
        runs = []
        for fold in folds:
            X_tr, y_tr, _, _, X_te, y_te = fold["automata"]
            runs.append((X_tr, y_tr, X_te, y_te))

    results = {}
    print(f"\n=== Parametre Tarama — {dataset} ===")
    print(f"{'Window':>8} {'Alphabet':>10} {'F1':>8} {'States':>8}")
    print("-" * 40)

    for window_size in window_sizes:
        for alphabet_size in alphabet_sizes:
            fold_f1s = []
            fold_states = []

            for X_tr, y_tr, X_te, y_te in runs:
                metrics = run_sweep_single(
                    X_tr, y_tr, X_te, y_te,
                    window_size, alphabet_size, seed
                )
                fold_f1s.append(metrics["f1"])
                fold_states.append(metrics["state_count"])

            key = f"w{window_size}_a{alphabet_size}"
            results[key] = {
                "window_size": window_size,
                "alphabet_size": alphabet_size,
                "mean_f1": float(np.mean(fold_f1s)),
                "std_f1": float(np.std(fold_f1s)),
                "mean_states": float(np.mean(fold_states)),
            }
            print(f"{window_size:>8} {alphabet_size:>10} {np.mean(fold_f1s):>8.4f} {np.mean(fold_states):>8.1f}")

    return results
