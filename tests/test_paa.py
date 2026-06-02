import numpy as np
import pytest
from src.models.paa import paa_transform


class TestPAATransform:
    def test_cikti_uzunlugu(self):
        series = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        result = paa_transform(series, n_segments=4)
        assert len(result) == 4

    def test_tek_segment(self):
        series = np.array([2.0, 4.0, 6.0, 8.0])
        result = paa_transform(series, n_segments=1)
        assert len(result) == 1
        assert np.isclose(result[0], np.mean(series))

    def test_segment_ortalamasi(self):
        series = np.array([1.0, 3.0, 5.0, 7.0])
        result = paa_transform(series, n_segments=2)
        assert np.isclose(result[0], 2.0)
        assert np.isclose(result[1], 6.0)

    def test_segment_sayisi_buyuk_hata(self):
        series = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            paa_transform(series, n_segments=5)

    def test_sabit_seri(self):
        series = np.ones(8) * 3.0
        result = paa_transform(series, n_segments=4)
        assert np.allclose(result, 3.0)
