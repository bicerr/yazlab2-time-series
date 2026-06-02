import pytest
import json
from src.models.automata import ProbabilisticAutomata
from src.explainability.explainer import AutomataExplainer
from src.explainability.formatter import to_json, to_table_row, explain_sequence


def build_explainer():
    automata = ProbabilisticAutomata()
    patterns = ["aab", "abc", "bcc", "abc", "aab", "abc", "bcc"]
    automata.fit(patterns)
    automata.get_transition_probs()
    return AutomataExplainer(automata)


class TestExplainerStep:
    def test_anahtarlar_tam(self):
        explainer = build_explainer()
        result = explainer.explain_step(["aab", "abc", "bcc", "abc"], time_step=0)
        for key in ["time_step", "state", "pattern", "status", "transitions",
                    "path_probability", "confidence_score", "decision"]:
            assert key in result

    def test_bilinen_pattern_status_known(self):
        explainer = build_explainer()
        result = explainer.explain_step(["aab", "abc", "bcc", "abc"], time_step=0)
        assert result["status"] == "known"

    def test_unseen_pattern_status_unseen(self):
        explainer = build_explainer()
        # İlk pattern unseen olmalı ki status unseen dönsün
        result = explainer.explain_step(["adc", "abc", "bcc", "abc"], time_step=0)
        assert result["status"] == "unseen"

    def test_path_probability_pozitif(self):
        explainer = build_explainer()
        result = explainer.explain_step(["aab", "abc", "bcc", "abc"], time_step=0)
        assert result["path_probability"] > 0

    def test_karar_anomaly_veya_normal(self):
        explainer = build_explainer()
        result = explainer.explain_step(["aab", "abc", "bcc", "abc"], time_step=0)
        assert result["decision"] in ["anomaly", "normal"]

    def test_confidence_path_prob_esit(self):
        explainer = build_explainer()
        result = explainer.explain_step(["aab", "abc", "bcc", "abc"], time_step=0)
        assert result["confidence_score"] == result["path_probability"]


class TestFormatter:
    def test_json_valid(self):
        explainer = build_explainer()
        result = explainer.explain_step(["aab", "abc", "bcc", "abc"], time_step=5)
        json_str = to_json(result)
        parsed = json.loads(json_str)
        assert parsed["time_step"] == 5
        assert "decision" in parsed

    def test_json_zorunlu_alanlar(self):
        explainer = build_explainer()
        result = explainer.explain_step(["aab", "abc", "bcc", "abc"], time_step=0)
        parsed = json.loads(to_json(result))
        for field in ["time_step", "state", "pattern", "status", "path_probability", "decision"]:
            assert field in parsed

    def test_tablo_satiri_string(self):
        explainer = build_explainer()
        result = explainer.explain_step(["aab", "abc", "bcc", "abc"], time_step=0)
        row = to_table_row(result)
        assert isinstance(row, str)
        assert "decision" in row

    def test_explain_sequence_uzunluk(self):
        explainer = build_explainer()
        patterns = ["aab", "abc", "bcc", "abc", "aab", "abc"]
        explanations = explain_sequence(explainer, patterns, window_size=4)
        assert len(explanations) == len(patterns) - 4 + 1
