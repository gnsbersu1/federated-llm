# =====================================================================
#  ADIM 2: BASELINE (merkezi eğitim, federe YOK)
#  Çalıştırma:  python baseline.py            (tam eğitim, ~5-10 dk)
#               python baseline.py --hizli    (10 adımlık deneme, kod çalışıyor mu diye)
#
#  Fikir: Tıp + finans verisinin HEPSİ (2000 soru) tek bir yerde olsaydı
#  ne kadar başarılı olurduk? Federe sonuçları bununla karşılaştıracağız.
#
#  Kaydedilenler (results/ klasörü):
#    baseline_loss.csv       her adımın eğitim loss'u (grafik için)
#    baseline_ozet.json      doğruluk ve test loss önce/sonra (alan alan), süre, LoRA boyutu
#    baseline_cevaplar.md    örnek sorularda modelin eğitim öncesi/sonrası seçtiği şık
#    baseline_adapter/       eğitilmiş LoRA ağırlıkları
# =====================================================================
import csv
import json
import os
import sys
import time

import config
import ortak

HIZLI = "--hizli" in sys.argv  # deneme modu: sadece 10 adım

ortak.seed_ayarla()
device = ortak.cihaz_sec()
print(f"Cihaz: {device}")
os.makedirs(config.SONUC_KLASORU, exist_ok=True)

# ---------------------------------------------------------------------
# 1) Model ve veri
# ---------------------------------------------------------------------
model, tok = ortak.model_yukle(device)
model.print_trainable_parameters()

# Baseline = iki istemcinin verisinin TOPLAMI (tıp 1000 + finans 1000 = 2000 soru)
egitim = ortak.jsonl_oku("istemciA_noniid") + ortak.jsonl_oku("istemciB_noniid")
test = ortak.jsonl_oku("test")
print(f"Eğitim: {len(egitim)} örnek, Test: {len(test)} örnek")

egitim_yukleyici = ortak.veri_yukleyici(egitim, tok, config.BATCH_BOYUTU, karistir=True)

# Eğitim öncesi/sonrası karşılaştıracağımız sabit sorular: her alandan ilk birkaç test sorusu
ornek_sorular = []
for alan in config.ALANLAR:
    ornek_sorular += [o for o in test if o["category"] == alan][: config.ORNEK_SORU_SAYISI]

# ---------------------------------------------------------------------
# 2) Eğitim ÖNCESİ ölçüm
# ---------------------------------------------------------------------
print(f"\nEğitim öncesi sınav: {len(test)} soru...")
once = ortak.alan_bazli_degerlendir(model, test, tok, device)
ortak.sonuc_yazdir("ÖNCE ", once)
cevaplar_once = [ortak.sik_tahmin_et(model, tok, o, device) for o in ornek_sorular]

# ---------------------------------------------------------------------
# 3) EĞİTİM
# ---------------------------------------------------------------------
print(f"\nEğitim başlıyor: {config.EPOCH} epoch, etkili batch {config.BATCH_BOYUTU * config.GRAD_BIRIKTIRME}, "
      f"~{len(egitim_yukleyici) * config.EPOCH // config.GRAD_BIRIKTIRME} adım")
kayit = []
t0 = time.time()
train_loss = ortak.egit(model, egitim_yukleyici, device, epoch=config.EPOCH,
                        max_adim=10 if HIZLI else None, kayit=kayit)
sure = time.time() - t0
print(f"Eğitim bitti: {sure / 60:.1f} dakika")

# ---------------------------------------------------------------------
# 4) Eğitim SONRASI ölçüm
# ---------------------------------------------------------------------
print("\nEğitim sonrası sınav...")
sonra = ortak.alan_bazli_degerlendir(model, test, tok, device)
print()
ortak.sonuc_yazdir("ÖNCE ", once)
ortak.sonuc_yazdir("SONRA", sonra)
cevaplar_sonra = [ortak.sik_tahmin_et(model, tok, o, device) for o in ornek_sorular]

# ---------------------------------------------------------------------
# 5) Kaydet
# ---------------------------------------------------------------------
on_ek = "baseline_hizli" if HIZLI else "baseline"
sonuc = os.path.join(config.SONUC_KLASORU, on_ek)

with open(f"{sonuc}_loss.csv", "w", newline="", encoding="utf-8") as f:
    yazici = csv.DictWriter(f, fieldnames=["adim", "epoch", "train_loss"])
    yazici.writeheader()
    yazici.writerows(kayit)

ozet = {
    "deney": "D1_baseline",
    "egitim_ornek": len(egitim),
    "adim": len(kayit),
    "dogruluk_once": {k: round(v, 1) for k, v in once["dogruluk"].items()},
    "dogruluk_sonra": {k: round(v, 1) for k, v in sonra["dogruluk"].items()},
    "test_loss_once": {k: round(v, 4) for k, v in once["loss"].items()},
    "test_loss_sonra": {k: round(v, 4) for k, v in sonra["loss"].items()},
    "ortalama_train_loss": round(train_loss, 4),
    "sure_dakika": round(sure / 60, 2),
    "lora_mb": round(ortak.lora_boyutu_mb(model), 2),
    "lora_r": config.LORA_R,
    "ogrenme_orani": config.OGRENME_ORANI,
    "etkili_batch": config.BATCH_BOYUTU * config.GRAD_BIRIKTIRME,
    "cihaz": device,
}
with open(f"{sonuc}_ozet.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=2, ensure_ascii=False)

with open(f"{sonuc}_cevaplar.md", "w", encoding="utf-8") as f:
    for i, (o, c_once, c_sonra) in enumerate(zip(ornek_sorular, cevaplar_once, cevaplar_sonra), 1):
        isaret = lambda c: "✓" if c == o["dogru"] else "✗"
        f.write(f"## Soru {i} ({o['category']})\n\n{o['messages'][0]['content']}\n\n")
        f.write(f"**Doğru cevap:** {o['messages'][1]['content']}\n\n")
        f.write(f"**Eğitim öncesi seçtiği:** {c_once} {isaret(c_once)}\n\n")
        f.write(f"**Eğitim sonrası seçtiği:** {c_sonra} {isaret(c_sonra)}\n\n---\n\n")

model.save_pretrained(f"{sonuc}_adapter")  # sadece LoRA ağırlıkları kaydedilir (birkaç MB)

print(f"\nSonuçlar kaydedildi: {sonuc}_*")
