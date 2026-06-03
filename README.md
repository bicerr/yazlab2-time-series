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
