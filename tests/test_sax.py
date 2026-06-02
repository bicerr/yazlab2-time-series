import numpy as np
import pytest
from src.models.sax import series_to_sax, get_breakpoints


class TestBreakpoints:
    def test_breakpoint_sayisi(self):
        bp = get_breakpoints(4)
        assert len(bp) == 3

    def test_breakpoint_siralanmis(self):
        bp = get_breakpoints(5)
        assert all(bp[i] < bp[i + 1] for i in range(len(bp) - 1))

    def test_iki_sinif_breakpoint(self):
        bp = get_breakpoints(2)
        assert len(bp) == 1
        assert np.isclose(bp[0], 0.0, atol=0.01)


class TestSeriesToSAX:
    def test_cikti_uzunlugu(self):
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = series_to_sax(series, n_segments=4, alphabet_size=3)
        assert len(result) == 4

    def test_sadece_kucuk_harf(self):
        series = np.random.randn(16)
        result = series_to_sax(series, n_segments=4, alphabet_size=4)
        assert all(c.islower() for c in result)

    def test_sabit_seri_tek_harf(self):
        series = np.zeros(8)
        result = series_to_sax(series, n_segments=4, alphabet_size=3)
        assert len(set(result)) == 1

    def test_alfabe_siniri(self):
        series = np.random.randn(16)
        alphabet_size = 4
        result = series_to_sax(series, n_segments=4, alphabet_size=alphabet_size)
        for ch in result:
            assert ord('a') <= ord(ch) < ord('a') + alphabet_size
