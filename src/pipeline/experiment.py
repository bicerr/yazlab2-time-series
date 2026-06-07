import numpy as np
from src.pipeline.pipeline import prepare_skab, prepare_batadal
from src.pipeline.dl_runner import run_dl_model
from src.pipeline.automata_runner import run_automata
from src.pipeline.noise import add_gaussian_noise
from src.pipeline.unseen import create_unseen_scenario
from src.evaluation.logger import load_logs, summarize_logs, log_experiment
from src.evaluation.statistical_tests import wilcoxon_test
from config.settings import cfg


SEEDS = cfg["experiment"]["seeds"]
SCENARIOS = ["original", "noisy", "unseen"]
DL_MODELS = ["LSTM", "GRU", "CNN"]


def run_batadal_experiments():
    print("\n=== BATADAL Deneyleri ===")
    batadal = prepare_batadal()

    for seed in SEEDS:
        print(f"\n-- Seed: {seed} --")
        for scenario in SCENARIOS:
            print(f"  Senaryo: {scenario}")

            dl_data = batadal["dl"]
            aut_data = batadal["automata"]

            X_tr, y_tr, X_val, y_val, X_te, y_te = dl_data
            X_tr_a, y_tr_a, X_val_a, y_val_a, X_te_a, y_te_a = aut_data

            if scenario == "noisy":
                X_te = add_gaussian_noise(X_te, seed=seed)
                X_te_a = add_gaussian_noise(X_te_a, seed=seed)

            elif scenario == "unseen":
                X_te_a, _, _, _ = create_unseen_scenario(X_tr_a, X_te_a, seed=seed)

            dl_data_run = (X_tr, y_tr, X_val, y_val, X_te, y_te)
            aut_data_run = (X_tr_a, y_tr_a, X_val_a, y_val_a, X_te_a, y_te_a)

            for model_name in DL_MODELS:
                result = run_dl_model(model_name, dl_data_run, "BATADAL", seed, scenario)
                print(f"    {model_name}: F1={result['metrics']['f1']:.4f}")

            result_aut = run_automata(aut_data_run, "BATADAL", seed, scenario)
            print(f"    Automata: F1={result_aut['metrics']['f1']:.4f}")


def run_skab_experiments():
    print("\n=== SKAB Deneyleri ===")
    skab_folds = prepare_skab()

    for seed in SEEDS:
        print(f"\n-- Seed: {seed} --")
        for scenario in SCENARIOS:
            print(f"  Senaryo: {scenario}")

            fold_results = {m: [] for m in DL_MODELS + ["Automata"]}

            for fold_idx, fold in enumerate(skab_folds):
                dl_data = fold["dl"]
                aut_data = fold["automata"]

                X_tr, y_tr, X_val, y_val, X_te, y_te = dl_data
                X_tr_a, y_tr_a, X_val_a, y_val_a, X_te_a, y_te_a = aut_data

                if scenario == "noisy":
                    X_te = add_gaussian_noise(X_te, seed=seed)
                    X_te_a = add_gaussian_noise(X_te_a, seed=seed)

                elif scenario == "unseen":
                    X_te_a, _, _, _ = create_unseen_scenario(X_tr_a, X_te_a, seed=seed)

                dl_data_run = (X_tr, y_tr, X_val, y_val, X_te, y_te)
                aut_data_run = (X_tr_a, y_tr_a, X_val_a, y_val_a, X_te_a, y_te_a)

                for model_name in DL_MODELS:
                    result = run_dl_model(model_name, dl_data_run, "SKAB", seed, scenario)
                    f1 = result["metrics"]["f1"]
                    fold_results[model_name].append(f1)
                    log_experiment(model_name, f"SKAB_fold{fold_idx}", seed, scenario, result["metrics"])

                result_aut = run_automata(aut_data_run, "SKAB", seed, scenario)
                f1_aut = result_aut["metrics"]["f1"]
                fold_results["Automata"].append(f1_aut)
                log_experiment("Automata", f"SKAB_fold{fold_idx}", seed, scenario, result_aut["metrics"])

            for model_name, f1_list in fold_results.items():
                mean_f1 = np.mean(f1_list)
                std_f1 = np.std(f1_list)
                print(f"    {model_name}: F1={mean_f1:.4f} ± {std_f1:.4f}")


def run_statistical_tests():
    """5 seed F1 skorları üzerinde Wilcoxon testi uygular."""
    logs = load_logs()

    def get_f1s(model, dataset, scenario):
        return [l["metrics"]["f1"] for l in logs
                if l["model"] == model and l["dataset"] == dataset
                and l["scenario"] == scenario]

    print("\n=== İstatistiksel Testler (Wilcoxon) ===")
    pairs = [("LSTM", "Automata"), ("GRU", "Automata"), ("CNN", "Automata")]
    for dataset in ["SKAB", "BATADAL"]:
        for m1, m2 in pairs:
            f1s_a = get_f1s(m1, dataset, "original")
            f1s_b = get_f1s(m2, dataset, "original")
            if len(f1s_a) >= 5 and len(f1s_b) >= 5:
                try:
                    result = wilcoxon_test(f1s_a[:5], f1s_b[:5])
                    print(f"  {m1} vs {m2} ({dataset}): p={result['p_value']:.4f} → {result['conclusion']}")
                except Exception as e:
                    print(f"  {m1} vs {m2} ({dataset}): test yapılamadı ({e})")


def run_all_experiments():
    run_batadal_experiments()
    run_skab_experiments()
    run_statistical_tests()
    logs = load_logs()
    summary = summarize_logs(logs)
    print("\n=== Özet ===")
    for key, val in summary.items():
        print(f"  {key}: mean_F1={val['mean_f1']:.4f} ± {val['std_f1']:.4f}")
    return summary


if __name__ == "__main__":
    run_all_experiments()
