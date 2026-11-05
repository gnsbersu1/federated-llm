# Deney Günlüğü

Her deneyden sonra buraya kısa bir not yaz. Raporun hammaddesi bu notlar olacak.

---

## D1 – Baseline (merkezi eğitim)

- **Tarih:** 3 Kasım 2026
- **Ne yaptım:** Tıp + finans sorularının hepsiyle (2000 soru), federe olmadan, tek makinede LoRA eğitimi
- **Ayarlar:** r=8, öğrenme oranı 2e-4, etkili batch 4, 1 epoch (500 adım)
- **Sonuç:**
  - Tıp doğruluk: önce %37,0 → sonra %37,5
  - Finans doğruluk: önce %58,0 → sonra %75,5
  - Süre: 3,8 dakika (NVIDIA RTX 4050)
  - LoRA boyutu: 2,16 MB

### Bu kısmı SEN dolduracaksın (sadece ___ yerlerini doldur)

- **Örnek sorularda ne değişti?** (results/baseline_cevaplar.md dosyasındaki ✓ işaretlerini say)
  - 6 örnek sorudan eğitim öncesi __4_ tanesi doğru, eğitim sonrası __4_ tanesi doğru.

- **Neden böyle olmuş olabilir?**
  - Finansta doğruluk arttı, çünkü olumsuz ya da olumluyu anlamak bir beceri ve eğitim setinde bu beceriyi kattı böylelikle doğruluk sayımız arttı  ___
  - Tıpta doğruluk artmadı,cunki finastaki gibi bir beceriden çok bir bilgi vardı  ___

---

## D2 – Federe öğrenme, non-IID (A = sadece tıp, B = sadece finans)

- **Tarih:** 4 Kasım 2026
- **Ne yaptım:** Flower simülasyonu, 2 istemci, 5 tur, her turda her istemci 50 adım (toplam 500 adım, baseline ile aynı). Sunucu FedAvg ile ortalama aldı. Veri istemcilerden hiç çıkmadı.
- **Sonuç (tur tur):**

| Tur | Tıp | Finans | Ağda giden toplam |
|---|---|---|---|
| 0 | %37,0 | %58,0 | 0 MB |
| 1 | %40,5 | %64,5 | 8,7 MB |
| 2 | %39,0 | %67,0 | 17,3 MB |
| 3 | %38,5 | %70,5 | 26,0 MB |
| 4 | %40,0 | %74,5 | 34,6 MB |
| 5 | %37,0 | %74,5 | 43,3 MB |

- **Baseline ile karşılaştırma:** Finans: federe %74,5, baseline %75,5. Tıp: federe %37,0, baseline %37,5.
- **Süre:** 5,4 dakika (baseline 3,8 dakika)
- **Tam model gönderilseydi:** ~1 GB × 4 transfer × 5 tur ≈ 20 GB. LoRA ile: 43 MB.

### Bu kısmı SEN dolduracaksın

- Hastane (A) hiç finans sorusu görmedi. Ortak model yine de finansta %74,5'e çıktı, çünkü ___
- Federe öğrenme, verileri tek yerde toplamaya (baseline) göre ___ (daha iyi / neredeyse aynı / çok daha kötü) sonuç verdi.
- Ağ trafiği açısından LoRA kullanmanın faydası: ___

---

## D3 – Federe öğrenme, IID (iki istemcide de karışık tıp + finans)

- **Tarih:** 5 Kasım 2026
- **Ne yaptım:** D2 ile aynı ayarlar. Tek fark: aynı 2000 soru karıştırılıp ikiye bölündü, her istemcide ~500 tıp + ~500 finans var.
- **Sonuç (tur tur):**

| Tur | Tıp | Finans | Ağda giden toplam |
|---|---|---|---|
| 0 | %37,0 | %58,0 | 0 MB |
| 1 | %37,0 | %63,0 | 8,7 MB |
| 2 | %37,0 | %63,0 | 17,3 MB |
| 3 | %36,5 | %74,5 | 26,0 MB |
| 4 | %35,0 | %73,5 | 34,6 MB |
| 5 | %36,5 | %74,0 | 43,3 MB |

- **Üç deneyin karşılaştırması (son durum):**

| | Tıp | Finans | Veri paylaşıldı mı? |
|---|---|---|---|
| D1 Baseline (merkezi) | %37,5 | %75,5 | Evet |
| D2 Federe non-IID | %37,0 | %74,5 | Hayır |
| D3 Federe IID | %36,5 | %74,0 | Hayır |

- **Süre:** 5,6 dakika
- **Not:** İlk denemede ekran kartı belleği dolduğu için deney çok yavaşladı (19 dakikada 1 tur bile bitmedi). Deneyi durdurup ekran kartını kullanan programları (Teams, Chrome, Spotify) kapattım ve yeniden başlattım.

### Bu kısmı SEN dolduracaksın

- IID ile non-IID arasında finansta ___ puan fark var. 200 soruda bu ___ soru eder.
- Bu yüzden şunu söyleyebilirim: verilerin istemciler arasında farklı dağılması (non-IID) bizim deneyimizde ___

---

## İleride yapılacak deneyler – unutma listesi

- [ ] **LoRA'yı küçült (r=8 → r=4):** LoRA dosyası yarıya iner, ağ trafiği de yarıya iner. Soru: doğruluk düşer mi?
- [ ] **Daha az tur:** Finans 4. turda zaten %74,5'e ulaşmıştı. Daha az turla aynı sonuç alınabilir mi? (Daha az tur = daha az trafik)

---

## Lab'da (3 Mac) yapılacaklar – unutma listesi

- [ ] Gerçek ağ trafiğini ölç ve buradaki hesapla (5 turda 43 MB) karşılaştır. Bilgisayarda simülasyon yaptığımız için MB'lar sadece hesaplandı, gerçekten ağdan geçmedi.
