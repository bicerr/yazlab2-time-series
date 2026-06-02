import numpy as np
import pytest
from src.models.automata import ProbabilisticAutomata


class TestAutomataFit:
    def setup_method(self):
        self.automata = ProbabilisticAutomata()
        self.patterns = ["abc", "bcd", "abc", "cde", "abc", "bcd"]
        self.automata.fit(self.patterns)

    def test_state_sayisi(self):
        assert len(self.automata.states) == 3

    def test_gecis_sayilari(self):
        assert self.automata.transition_counts["abc"]["bcd"] == 2
        assert self.automata.transition_counts["abc"]["cde"] == 1

    def test_bos_pattern_listesi(self):
        a = ProbabilisticAutomata()
        a.fit([])
        assert len(a.states) == 0


class TestTransitionProbs:
    def setup_method(self):
        self.automata = ProbabilisticAutomata()
        self.patterns = ["abc", "bcd", "abc", "bcd", "abc"]
        self.automata.fit(self.patterns)
        self.probs = self.automata.get_transition_probs(laplace_alpha=1.0)

    def test_olasilik_toplami(self):
        for state in self.automata.states:
            total = sum(self.probs[state].values())
            assert np.isclose(total, 1.0, atol=1e-6)

    def test_laplace_sifir_olasilik_yok(self):
        for state in self.automata.states:
            for next_state in self.automata.states:
                assert self.probs[state][next_state] > 0


class TestPathProbability:
    def setup_method(self):
        self.automata = ProbabilisticAutomata()
        patterns = ["abc", "bcd", "abc", "bcd", "cde", "abc"]
        self.automata.fit(patterns)
        self.automata.get_transition_probs()

    def test_bilinen_path_pozitif(self):
        prob = self.automata.compute_path_probability(["abc", "bcd"])
        assert prob > 0

    def test_tek_pattern_prob_bir(self):
        prob = self.automata.compute_path_probability(["abc"])
        assert np.isclose(prob, 1.0)

    def test_unseen_path_cok_kucuk(self):
        prob = self.automata.compute_path_probability(["xyz", "zzz"])
        assert prob < 1e-5
