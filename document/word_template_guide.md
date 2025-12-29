# 📄 Akademik Rapor Şablonu

Bu dosya, fraud detection projesi içeriğini akademik bir Word (.docx) raporuna nasıl eklemeniz gerektiğini göstermektedir.

---

## 📋 Önerilen Word Doküman Yapısı

### Sayfa Ayarları
- **Sayfa Boyutu**: A4
- **Kenar Boşlukları**: Üst/Alt: 2.5 cm, Sol/Sağ: 3 cm
- **Yazı Tipi**: Times New Roman veya Arial
- **Yazı Boyutu**: Metin: 12pt, Başlıklar: 14-16pt
- **Satır Aralığı**: 1.5

---

## 📑 Bölüm Yapısı

### 1. Kapak Sayfası
```
[Üniversite Logosu]

PROJE BAŞLIĞI:
"Makine Öğrenmesi Tabanlı Banka Dolandırıcılık Tespit Sistemi"

Hazırlayan:
[Adınız Soyadınız]
[Öğrenci No]

Danışman:
[Prof./Doç./Dr. Adı Soyadı]

[Tarih: Aralık 2025]
```

---

### 2. Özet / Abstract (Yarım sayfa)
```
ÖZET

Bu projede, banka para transferlerinde dolandırıcılık tespiti için 
XGBoost tabanlı bir makine öğrenmesi modeli geliştirilmiştir. Sistem, 
16 türetilmiş öznitelik kullanarak gerçek zamanlı risk değerlendirmesi 
yapmaktadır. Model, AUC-PR: 0.95, AUC-ROC: 0.98 ve F1 Score: 0.90 
performans değerlerine ulaşmıştır.

Anahtar Kelimeler: Dolandırıcılık Tespiti, XGBoost, Makine Öğrenmesi, 
Feature Engineering, Risk Değerlendirmesi
```

---

### 3. İçindekiler (Otomatik - Word'de Table of Contents)
```
İÇİNDEKİLER

1. Giriş ..................................................... 1
2. Metodoloji ................................................ 3
   2.1 Model Mimarisi ........................................ 3
   2.2 Öznitelik Mühendisliği ................................ 4
   2.3 Preprocessing Pipeline ................................ 6
3. Sonuçlar .................................................. 8
   3.1 Performans Metrikleri ................................. 8
   3.2 Tespit Edilen Dolandırıcılık Desenleri ................ 9
4. Tartışma ve Sonuç ......................................... 11
Kaynaklar .................................................... 12
```

---

### 4. Giriş (1-2 sayfa)
```
1. GİRİŞ

Günümüzde dijital bankacılık hizmetlerinin yaygınlaşmasıyla birlikte, 
çevrimiçi dolandırıcılık olayları önemli bir güvenlik tehdidi haline 
gelmiştir. Bu çalışmada, para transferi işlemlerini gerçek zamanlı 
olarak değerlendiren ve potansiyel dolandırıcılık girişimlerini tespit 
eden bir makine öğrenmesi sistemi geliştirilmiştir.

1.1 Problem Tanımı
... [Dolandırıcılık problemini açıklayın]

1.2 Amaç ve Hedefler
... [Projenin amaçlarını listeleyin]

1.3 Kapsam
... [Çalışmanın kapsamını belirtin]
```

---

### 5. Metodoloji (3-4 sayfa)

#### 5.1 Model Mimarisi
```
2. METODOLOJİ

2.1 Model Mimarisi

Bu çalışmada XGBoost (eXtreme Gradient Boosting) sınıflandırıcısı 
kullanılmıştır. Tablo 1'de model hiperparametreleri verilmiştir.

| Parametre      | Değer          |
|----------------|----------------|
| n_estimators   | 400            |
| learning_rate  | 0.03           |
| max_depth      | 6              |
| subsample      | 0.85           |
| eval_metric    | AUC-PR         |

Tablo 1: XGBoost Model Hiperparametreleri
```

#### 5.2 Feature Engineering (Görsel 1 Buraya)
```
2.2 Öznitelik Mühendisliği

Ham işlem verilerinden toplam 16 öznitelik türetilmiştir. Şekil 1'de 
öznitelik kategorileri gösterilmektedir.

[academic_features.png GÖRSELİ BURAYA - "Figure 1: Feature Categories"]

Şekil 1: Öznitelik Kategorileri

Öznitelikler dört kategoride gruplandırılmıştır:
- Oran Öznitelikleri (5 adet)
- Kart Öznitelikleri (2 adet)
- Risk Göstergeleri (7 adet)
- Zaman Öznitelikleri (2 adet)
```

#### 5.3 Preprocessing Pipeline (Görsel 4 Buraya)
```
2.3 Preprocessing Pipeline

Veri ön işleme süreci Şekil 2'de gösterilmektedir.

[academic_pipeline.png GÖRSELİ BURAYA - "Figure 4: Pipeline"]

Şekil 2: Preprocessing Pipeline Mimarisi

Numerik öznitelikler için StandardScaler, kategorik öznitelikler 
için OneHotEncoder kullanılmıştır.
```

---

### 6. Sonuçlar (2-3 sayfa)

#### 6.1 Performans Metrikleri (Görsel 2 Buraya)
```
3. SONUÇLAR

3.1 Performans Metrikleri

Model değerlendirme sonuçları Şekil 3'te ve Tablo 2'de verilmiştir.

[academic_metrics.png GÖRSELİ BURAYA - "Figure 2: Metrics"]

Şekil 3: Model Performans Metrikleri

| Metrik   | Değer  |
|----------|--------|
| AUC-PR   | 0.95   |
| AUC-ROC  | 0.98   |
| F1 Score | 0.90   |

Tablo 2: Model Performans Değerleri
```

#### 6.2 Fraud Patterns (Görsel 3 Buraya)
```
3.2 Tespit Edilen Dolandırıcılık Desenleri

Şekil 4'te tespit edilen 5 temel dolandırıcılık deseni gösterilmektedir.

[academic_patterns.png GÖRSELİ BURAYA - "Figure 3: Patterns"]

Şekil 4: Dolandırıcılık Desen Dağılımı
```

#### 6.3 Karar Mekanizması (Görsel 5 Buraya)
```
3.3 Risk Değerlendirme ve Karar Mekanizması

Şekil 5'te risk değerlendirme karar akışı gösterilmektedir.

[academic_decision.png GÖRSELİ BURAYA - "Figure 5: Decision"]

Şekil 5: Risk Değerlendirme Karar Akışı

| Risk Skoru | Seviye | Aksiyon  |
|------------|--------|----------|
| 0.00-0.30  | LOW    | APPROVE  |
| 0.30-0.70  | MEDIUM | HOLD     |
| 0.70-1.00  | HIGH   | BLOCK    |

Tablo 3: Risk Seviyeleri ve Aksiyonlar
```

---

### 7. Tartışma ve Sonuç (1 sayfa)
```
4. TARTIŞMA VE SONUÇ

Bu çalışmada geliştirilen dolandırıcılık tespit sistemi, AUC-PR 0.95 
gibi yüksek bir performans sergilemiştir. Sistem, 5 farklı dolandırıcılık 
desenini başarıyla tanımlamakta ve gerçek zamanlı risk değerlendirmesi 
yapabilmektedir.

4.1 Katkılar
- 16 türetilmiş öznitelik ile ölçek-bağımsız feature engineering
- Açıklanabilir kararlar (SHAP feature importance)
- Kafka ile yüksek throughput stream processing desteği

4.2 Gelecek Çalışmalar
- Deep learning modellerinin entegrasyonu
- Çevrimiçi öğrenme yeteneklerinin eklenmesi
- Farklı dolandırıcılık türlerinin tespiti
```

---

### 8. Kaynaklar
```
KAYNAKLAR

[1] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting 
    system. In Proceedings of KDD '16 (pp. 785-794).

[2] Pozzolo, A. D., et al. (2015). Credit card fraud detection: A
    realistic modeling and a novel learning strategy. IEEE TNNLS.

[3] Sklearn Documentation. (2024). Preprocessing data. 
    https://scikit-learn.org/stable/modules/preprocessing.html

[4] FastAPI Documentation. (2024). FastAPI framework.
    https://fastapi.tiangolo.com/
```

---

## 📌 Word'e Eklerken Dikkat Edilecekler

### Görseller İçin:
1. **Ekle → Resim** ile görselleri ekleyin
2. Her görselin altına başlık ekleyin: "Şekil X: Başlık"
3. Görselleri **metin içine** değil **satır/paragraf** olarak yerleştirin
4. Görsellerin boyutunu sayfa genişliğinin %80-90'ı olacak şekilde ayarlayın

### Tablolar İçin:
1. **Ekle → Tablo** kullanın
2. Her tablonun üstüne veya altına başlık ekleyin: "Tablo X: Başlık"
3. Kenarlıkları basit tutun

### Başlıklar İçin:
1. Word'de **Heading 1, Heading 2, Heading 3** stillerini kullanın
2. Bu sayede otomatik İçindekiler oluşturabilirsiniz
3. İçindekiler için: **Başvurular → İçindekiler → Otomatik**

### Sayfa Numaraları:
1. **Ekle → Sayfa Numarası → Sayfa Altı → Ortalanmış**

---

## 📁 Görsel Dosyaları

Aşağıdaki görselleri Word'e sırayla ekleyebilirsiniz:

| Dosya Adı | Şekil No | Kullanım Yeri |
|-----------|----------|---------------|
| academic_features*.png | Şekil 1 | Feature Engineering bölümü |
| academic_metrics*.png | Şekil 3 | Performans Metrikleri bölümü |
| academic_patterns*.png | Şekil 4 | Fraud Patterns bölümü |
| academic_pipeline*.png | Şekil 2 | Preprocessing bölümü |
| academic_decision*.png | Şekil 5 | Karar Mekanizması bölümü |

---

*Bu şablon, akademik rapor yazım standartlarına uygun olarak hazırlanmıştır.*
