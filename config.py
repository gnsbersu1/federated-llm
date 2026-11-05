# =====================================================================
#  Projenin TÜM ayarları burada, tek bir yerde.
#  Neden tek yer? Sunucu ve istemciler LoRA'yı farklı ayarlarla kurarsa
#  ağırlıkların boyutları uyuşmaz ve "shape mismatch" hatası çıkar.
#  Bir ayarı değiştireceksen sadece bu dosyayı değiştir.
# =====================================================================
import os

# --- Model ---
MODEL_ADI = "Qwen/Qwen2.5-0.5B-Instruct"  # Hocanın istediği Qwen; küçük sürüm (~0.5 milyar parametre)

# --- LoRA ayarları ---
LORA_R = 8                # rank: eklenen küçük matrislerin "genişliği". Büyüdükçe daha güçlü ama daha çok MB
#  alpha (LoRA güncellemesinin etki çarpanı) otomatik olarak 2 × r alınır → r=8 için 16
LORA_DROPOUT = 0.05       # eğitimde rastgele %5'i kapatır; ezberlemeyi azaltır
LORA_HEDEF_KATMANLAR = ["q_proj", "v_proj"]  # dikkat (attention) katmanlarının sorgu ve değer parçaları

# --- Veri ---
# İki farklı ALAN (konu). Senaryo: İstemci A bir hastane, İstemci B bir finans şirketi.
# Her biri kendi alanının verisine sahip ve bu veriyi dışarı veremez.
# İki alanda da sorular ŞIKLI ve doğru cevap belli → başarıyı "% kaç doğru" diye ölçebiliriz.
ALANLAR = {
    "tip": "openlifescienceai/medmcqa",          # Hindistan tıp sınavı soruları, 4 şık (A-D)
    "finans": "FinGPT/fingpt-sentiment-train",   # finans haberi: olumsuz / nötr / olumlu, 3 şık (A-C)
}
NONIID_ISTEMCI_ALANI = {"A": "tip", "B": "finans"}  # non-IID'de kim hangi alanı görür

ALAN_BASINA_EGITIM = 1000    # her alandan kaç eğitim örneği → toplam 2000
ALAN_BASINA_TEST = 200       # her alandan kaç test (sınav) sorusu → toplam 400
MAX_TOKEN = 512              # bir örneğin en fazla kaç token olacağı (daha uzunlar atılır)
ISTEMCI_SAYISI = 2           # İstemci A ve İstemci B

# --- Eğitim ---
OGRENME_ORANI = 2e-4    # her adımda ağırlıkların ne kadar değişeceği
BATCH_BOYUTU = 2        # aynı anda kaç örnek işleneceği (6 GB ekran kartında 4 sığmıyor)
GRAD_BIRIKTIRME = 2     # kaç batch'in düzeltmesi biriktirilip birlikte uygulanacağı
                        # → etkili batch = 2 × 2 = 4 örnek (bellek az, sonuç batch 4 ile aynı)
SEED = 42               # rastgelelik tohumu: aynı sayı = aynı sonuç (deneyleri karşılaştırabilmek için)
EPOCH = 1               # baseline'da tüm eğitim verisinin üzerinden kaç kez geçileceği
ORNEK_SORU_SAYISI = 3   # her alandan, eğitim öncesi/sonrası cevaplarını karşılaştıracağımız soru sayısı

# --- Federe öğrenme (Flower) ---
TUR_SAYISI = 5          # sunucu kaç tur "dağıt → eğit → topla → ortala" yapacak
LOKAL_ADIM = 50         # her turda her istemci kaç adım eğitecek (50 adım × 4 örnek = 200 soru)
# Adil karşılaştırma: 5 tur × 50 adım = 250 adım/istemci; 2 istemci → toplam 500 adım.
# Baseline da 500 adım eğitmişti. Yani iki yöntem de AYNI miktarda eğitim yapıyor.

# --- Klasörler ---
# Veri ve sonuçlar her makinede SABİT bir yerde durur: kullanıcı klasörü/fedlora_proje
#   Windows: C:\Users\<kullanıcı>\fedlora_proje      Mac: /Users/<kullanıcı>/fedlora_proje
# Neden proje klasöründe değil? Flower uygulamayı çalıştırırken kodu başka bir yere kopyalar,
# veriyi kopyalamaz. Federe öğrenmenin mantığı da bu: veri kodla birlikte taşınmaz,
# her makinenin verisi o makinede kendi yerinde durur.
CALISMA_KLASORU = os.path.join(os.path.expanduser("~"), "fedlora_proje")
VERI_KLASORU = os.path.join(CALISMA_KLASORU, "data")
SONUC_KLASORU = os.path.join(CALISMA_KLASORU, "results")
