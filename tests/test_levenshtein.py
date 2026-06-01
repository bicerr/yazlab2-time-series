import pytest
from src.models.levenshtein import levenshtein_distance, find_nearest_pattern
from src.models.automata import ProbabilisticAutomata


class TestLevenshteinDistance:
    def test_ayni_string(self):
        assert levenshtein_distance("abc", "abc") == 0

    def test_tek_karakter_fark(self):
        assert levenshtein_distance("abc", "adc") == 1

    def test_tamamen_farkli(self):
        assert levenshtein_distance("abc", "xyz") == 3

    def test_bos_string(self):
        assert levenshtein_distance("", "abc") == 3
        assert levenshtein_distance("abc", "") == 3

    def test_farkli_uzunluk(self):
        assert levenshtein_distance("ab", "abc") == 1

    def test_simetrik(self):
        assert levenshtein_distance("abc", "bca") == levenshtein_distance("bca", "abc")


class TestFindNearestPattern:
    def test_en_yakin_bulunur(self):
        vocab = {"abc", "aab", "bcc"}
        nearest, dist = find_nearest_pattern("adc", vocab)
        assert nearest == "abc"
        assert dist == 1

    def test_tam_esleme(self):
        vocab = {"abc", "aab", "bcc"}
        nearest, dist = find_nearest_pattern("abc", vocab)
        assert nearest == "abc"
        assert dist == 0

    def test_bos_sozluk_hata_verir(self):
        with pytest.raises(ValueError):
            find_nearest_pattern("abc", set())


class TestUnseenPatternAutomata:
    def setup_method(self):
        self.automata = ProbabilisticAutomata()
        patterns = ["aab", "abc", "bcc", "abc", "aab", "abc"]
        self.automata.fit(patterns)
        self.automata.get_transition_probs()

    def test_bilinen_pattern_known_dondutur(self):
        resolved, status, mapped_to, dist = self.automata.resolve_pattern("abc")
        assert status == "known"
        assert resolved == "abc"
        assert dist == 0

    def test_unseen_pattern_map_edilir(self):
        resolved, status, mapped_to, dist = self.automata.resolve_pattern("adc")
        assert status == "unseen"
        assert mapped_to is not None
        assert dist > 0

    def test_unseen_pattern_tahmin_yapar(self):
        patterns = ["adc", "abc", "bcc", "aab", "abc"]
        preds, probs = self.automata.predict(patterns)
        assert len(preds) > 0
        assert all(p in [0, 1] for p in preds)
