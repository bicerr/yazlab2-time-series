# From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

Bu proje, zaman serisi anomali tespitinde derin öğrenme modellerini (LSTM, GRU, 1D-CNN) olasılıksal otomata tabanlı bir yaklaşımla karşılaştırmakta ve otomata modelinin açıklanabilirlik avantajını ortaya koymaktadır.

---

## Gereksinimler

- Python 3.10+
- Aşağıdaki komutla bağımlılıklar kurulabilir:

```bash
pip install -r requirements.txt
```

**Kullanılan başlıca kütüphaneler:**

| Kütüphane | Amaç |
|-----------|------|
| PyTorch | LSTM, GRU, 1D-CNN modelleri |
| scikit-learn | Ön işleme, metrikler, GroupKFold |
| scipy | İstatistiksel testler (Wilcoxon) |
| numpy / pandas | Veri işleme |
| matplotlib / seaborn | Görselleştirme |
| pytest | Birim testler |

---

## Kurulum ve Çalıştırma

```bash
# Repoyu klonla
git clone https://github.com/kullanici/yazlab2-time-series.git
cd yazlab2-time-series

# Bağımlılıkları kur
pip install -r requirements.txt

# Testleri çalıştır
python -m pytest tests/ -v

# Pipeline'ı başlat
python -m src.pipeline.pipeline
```

---

## Veri Setleri

Proje iki farklı veri seti üzerinde çalışmaktadır:

- **SKAB** (Skoltech Anomaly Benchmark): Endüstriyel sensör verisi, `valve1` ve `valve2` gruplarından oluşur. Değerlendirme için `GroupKFold` (5-fold) kullanılmıştır.
- **BATADAL** (Battle of the Attack Detection Algorithms): Su dağıtım sistemi saldırı verisi. Zaman sıralı %60/%20/%20 train/val/test bölmesi kullanılmıştır.

Veri dosyaları `data/raw/` dizinine yerleştirilmelidir:
```
data/
  raw/
    skab/
      valve1/
      valve2/
    batadal/
```

---

## Proje Mimarisi

```
src/
├── preprocessing/
│   ├── loader.py          # SKAB ve BATADAL veri yükleyici
│   └── preprocessor.py    # Normalizasyon, PCA, train/val/test bölme
├── models/
│   ├── lstm_model.py      # LSTM mimarisi
│   ├── gru_model.py       # GRU mimarisi
│   ├── cnn_model.py       # 1D-CNN mimarisi
│   ├── trainer.py         # Eğitim döngüsü (early stopping)
│   ├── predictor.py       # Tahmin fonksiyonları
│   ├── paa.py             # Piecewise Aggregate Approximation
│   ├── sax.py             # Symbolic Aggregate approXimation
│   ├── pattern_extractor.py  # Sliding window pattern çıkarımı
│   ├── automata.py        # Olasılıksal otomata (Laplace smoothing)
│   └── levenshtein.py     # Unseen pattern yönetimi
├── explainability/
│   ├── explainer.py       # State, pattern, path probability, güven skoru
│   └── formatter.py       # JSON ve tablo formatında çıktı
├── evaluation/
│   ├── metrics.py         # Accuracy, Precision, Recall, F1
│   ├── statistical_tests.py  # Wilcoxon ve McNemar testleri
│   ├── logger.py          # JSON deney loglama sistemi
│   ├── visualizer.py      # Confusion matrix, ROC, PR, duyarlılık grafikleri
│   └── automata_viz.py    # Otomata durum diyagramı ve geçiş heatmap
└── pipeline/
    ├── pipeline.py        # Ana pipeline
    ├── dl_runner.py       # Derin öğrenme model koşucusu
    ├── automata_runner.py # Otomata koşucusu
    ├── experiment.py      # 3 senaryo × 5 seed deney döngüsü
    ├── noise.py           # Gaussian gürültü ekleme
    ├── unseen.py          # Unseen pattern senaryo üretici
    ├── cross_dataset.py   # Çapraz veri seti deneyleri
    └── parameter_sweep.py # Window/alphabet parametre tarama
```

---

## Modeller

### Derin Öğrenme Modelleri
- **LSTM** (Long Short-Term Memory): 2 katmanlı, hidden size 64, dropout 0.2
- **GRU** (Gated Recurrent Unit): 2 katmanlı, hidden size 64, dropout 0.2
- **1D-CNN**: 2 konvolüsyon katmanı (64→128 filtre), AdaptiveAvgPool, dropout 0.2

Tüm modeller sliding window sekansları üzerinde eğitilmekte, erken durdurma (early stopping) ile aşırı öğrenme önlenmektedir.

### Olasılıksal Otomata
1. Ham zaman serisi → **PAA** ile segment ortalamalarına indirgenir
2. PAA çıktısı → **SAX** ile sembolik diziye dönüştürülür
3. Sliding window ile **pattern dizileri** çıkarılır
4. **Geçiş matrisi** Laplace smoothing ile hesaplanır
5. **Path probability** eşik altındaysa anomali kararı verilir
6. Görülmemiş patternler **Levenshtein mesafesi** ile en yakın bilinen patterne eşlenir
