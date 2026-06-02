import numpy as np
import pytest
from src.preprocessing.preprocessor import fit_transform_scaler, fit_transform_pca, split_batadal, split_skab


class TestScaler:
    def test_cikti_sekli(self):
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(20, 5)
        X_train_s, _, X_test_s, scaler = fit_transform_scaler(X_train, X_test=X_test)
        assert X_train_s.shape == X_train.shape
        assert X_test_s.shape == X_test.shape

    def test_train_ortalama_sifir(self):
        X_train = np.random.randn(100, 5)
        X_train_s, _, _, _ = fit_transform_scaler(X_train)
        assert np.allclose(X_train_s.mean(axis=0), 0.0, atol=1e-6)

    def test_val_none(self):
        X_train = np.random.randn(50, 3)
        _, X_val_s, _, _ = fit_transform_scaler(X_train)
        assert X_val_s is None


class TestPCA:
    def test_boyut_azalir(self):
        X_train = np.random.randn(100, 10)
        X_train_p, _, _, _ = fit_transform_pca(X_train)
        assert X_train_p.shape[1] < X_train.shape[1]

    def test_cikti_sekli(self):
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(20, 5)
        X_train_p, _, X_test_p, _ = fit_transform_pca(X_train, X_test=X_test)
        assert X_train_p.shape[0] == 100
        assert X_test_p.shape[0] == 20


class TestSplitBatadal:
    def test_bolme_oranlari(self):
        X = np.random.randn(100, 3)
        y = np.zeros(100)
        (X_tr, _), (X_val, _), (X_te, _) = split_batadal(X, y)
        assert len(X_tr) == 60
        assert len(X_val) == 20
        assert len(X_te) == 20

    def test_toplam_satir(self):
        X = np.random.randn(100, 3)
        y = np.zeros(100)
        (X_tr, _), (X_val, _), (X_te, _) = split_batadal(X, y)
        assert len(X_tr) + len(X_val) + len(X_te) == 100


class TestSplitSkab:
    def test_fold_sayisi(self):
        X = np.random.randn(50, 3)
        y = np.array([0] * 25 + [1] * 25)
        groups = np.array([f"file{i % 10}.csv" for i in range(50)])
        folds = split_skab(X, y, groups)
        assert len(folds) == 5

    def test_fold_indeks_tipi(self):
        X = np.random.randn(50, 3)
        y = np.array([0] * 25 + [1] * 25)
        groups = np.array([f"file{i % 10}.csv" for i in range(50)])
        folds = split_skab(X, y, groups)
        for fold in folds:
            assert len(fold) >= 2
            assert len(fold[0]) > 0
            assert len(fold[1]) > 0
