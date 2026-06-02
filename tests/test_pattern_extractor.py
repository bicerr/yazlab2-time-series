import numpy as np
import pytest
from src.models.pattern_extractor import extract_patterns, build_sax_vocabulary


class TestExtractPatterns:
    def test_pattern_sayisi(self):
        X = np.random.randn(20, 1)
        patterns = extract_patterns(X, window_size=4, n_segments=4, alphabet_size=3)
        assert len(patterns) == 20 - 4 + 1

    def test_pattern_tipi_string(self):
        X = np.random.randn(10, 1)
        patterns = extract_patterns(X, window_size=3, n_segments=3, alphabet_size=3)
        assert all(isinstance(p, str) for p in patterns)

    def test_pattern_uzunlugu(self):
        X = np.random.randn(10, 1)
        n_segments = 3
        patterns = extract_patterns(X, window_size=3, n_segments=n_segments, alphabet_size=3)
        assert all(len(p) == n_segments for p in patterns)

    def test_tek_ornek(self):
        X = np.ones((5, 1))
        patterns = extract_patterns(X, window_size=5, n_segments=4, alphabet_size=3)
        assert len(patterns) == 1


class TestBuildSAXVocabulary:
    def test_sozluk_benzersiz(self):
        patterns = ["abc", "abc", "bcd", "abc", "bcd"]
        vocab = build_sax_vocabulary(patterns)
        assert vocab == {"abc", "bcd"}

    def test_bos_liste(self):
        vocab = build_sax_vocabulary([])
        assert vocab == set()

    def test_sozluk_tipi(self):
        patterns = ["abc", "bcd"]
        vocab = build_sax_vocabulary(patterns)
        assert isinstance(vocab, set)
