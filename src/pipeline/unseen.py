import numpy as np
from src.models.pattern_extractor import extract_patterns, build_sax_vocabulary
from src.models.sax import series_to_sax
from config.settings import cfg


def create_unseen_scenario(X_train: np.ndarray, X_test: np.ndarray, seed: int = 42):
    """
    Test verisinde eğitim sözlüğünde bulunmayan pattern'lar üretir.
    Yöntem: Daha büyük alphabet_size kullanarak sözlük uzayını genişletir.
    Bu sayede test pattern'larının bir kısmı doğal olarak unseen olur.
    PDF: 'SAX sözlüğünde bulunmayan pattern'lar unseen olarak kabul edilir'
    """
    np.random.seed(seed)
    window_size = cfg["automata"]["window_size"]

    # Unseen senaryosu için daha büyük alfabe (daha fazla olası pattern)
    unseen_alphabet_size = 6  # 6^4 = 1296 olası pattern → doğal unseen garantisi

    # Train sözlüğünü büyük alfabe ile oluştur
    train_series = X_train.flatten()
    train_patterns = extract_patterns(
        train_series,
        window_size=window_size,
        alphabet_size=unseen_alphabet_size
    )
    vocabulary = build_sax_vocabulary(train_patterns)

    # Test pattern'larını aynı alfabe ile çıkar
    test_series = X_test.flatten()
    test_patterns = extract_patterns(
        test_series,
        window_size=window_size,
        alphabet_size=unseen_alphabet_size
    )

    unseen_count = sum(1 for p in test_patterns if p not in vocabulary)
    total = len(test_patterns)
    print(f"Unseen pattern oranı: {unseen_count}/{total} ({100*unseen_count/total:.1f}%)")

    return X_test, vocabulary, test_patterns, unseen_alphabet_size
