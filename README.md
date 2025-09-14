## 📦 Shipping Data Anomaly Detection (Autoencoder)

**Veri:** `shippingdata.csv` (10.999 satır, 12 değişken)  
**Hedef değişken:** `Reached.on.Time_Y.N` (1 = gecikme)

---

### 🔹 Yaklaşım
- Sayısal + one-hot kategorik değişkenler **StandardScaler** ile ölçeklendi  
- **Autoencoder (AE)** MSE rekonstrüksiyon kaybıyla eğitildi  
- Her kayıt için **`recon_error`** anomali skoru olarak kullanıldı  
- Eğitim Mimarisi: Dense 64 → … → 64 (sıkıştır–aç), aktivasyon ReLU, kayıp MSE, optimizer Adam  
- Loss/Val_Loss birlikte ve düzenli azaldı → **Overfitting yok**  

---

### 🔹 Eşikleme (Thresholding)
- Başlangıç: üst %5 dilim anomali  
- Optimizasyon: %70–99.5 yüzdelik tarandı, sınıf 1 (gecikme) için **F1 maksimize edildi**  
- En iyi eşik: **%80 (≈ 0.567 recon_error)**  

---

### 🔹 Sonuçlar (En iyi eşikte)
- Precision (class=1): **0.68**  
- Recall (class=1): **0.23**  
- F1 (class=1): **0.34**  
- Accuracy: **0.48**

Model, gecikenleri işaretlerken daha isabetli (precision yüksek) ama azını yakalıyor (recall düşük). Normal kayıtları ayırmada daha güçlü.  

---

### 🔹 Görselleştirmeler
- **Histogram:** Recon_error çoğunlukla düşük; eşik sağında kalan küçük grup anomaliler  
- **Confusion Matrix:** Gecikmelerin önemli bölümü kaçırılıyor (low recall)  
- **Boxplot:** Anomali etiketlerinde hata dağılımı belirgin şekilde daha yüksek  

---

### 🔹 Sınırlamalar
- **Unsupervised AE** → etiket kullanmıyor  
- Gecikme sınıfı dengesiz → recall düşük  
- Zaman bilgisinin olmaması → LSTM/RNN benzeri dizisel modeller uygulanamıyor  

---

### 🔹 İyileştirme Önerileri (Fine-Tuning / Next)
- Daha derin AE, BatchNorm + Dropout, L2, EarlyStopping/ReduceLROnPlateau  
- Özellik mühendisliği: etkileşim terimleri (ağırlık/indirim, maliyet/mesafe), log-dönüşümler  
- Eşik stratejisi: İş maliyetine göre precision–recall trade-off  
- Alternatifler: Isolation Forest / One-Class SVM / Random Forest (denetimli, etiketli model)  
- Veri: Zaman damgası eklenirse **LSTM/RNN** ile gecikme olasılığı tahmini yapılabilir  

---
