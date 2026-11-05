# =====================================================================
#  SUNUCU (ServerApp)
#
#  Sunucunun elinde HİÇ veri yok. Yaptığı iş:
#    1. Boş (henüz bir şey öğrenmemiş) LoRA ağırlıklarını hazırla
#    2. Her turda: istemcilere dağıt → eğitsinler → geri topla → ortalamasını al (FedAvg)
#    3. Her turun sonunda ortak modeli sınava sok (tıp ve finans doğruluğu)
#    4. Tur başına ağda kaç MB gidip geldiğini hesapla
#
#  Sınav (test) soruları sadece ÖLÇMEK için. Eğitimde hiç kullanılmıyor.
# =====================================================================
import csv
import os
import time

import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from peft import get_peft_model_state_dict, set_peft_model_state_dict

import config
import ortak

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    bolme = context.run_config["bolme"]
    tur_sayisi = context.run_config["tur-sayisi"]
    device = ortak.cihaz_sec()
    test = ortak.jsonl_oku("test")
    print(f"\n=== FEDERE EĞİTİM: bölme={bolme}, {tur_sayisi} tur, 2 istemci ===")

    # --- 1) Başlangıç LoRA'sı (boş: modele henüz hiçbir etkisi yok) ---
    ortak.seed_ayarla()
    model, _ = ortak.model_yukle(device)
    baslangic = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
    del model
    lora_mb = sum(v.numel() * v.element_size() for v in baslangic.values()) / 1e6

    kayitlar = []           # her turun sonuçları (CSV'ye yazılacak)
    baslama = time.time()

    def sinav(tur: int, arrays: ArrayRecord) -> MetricRecord:
        """Her tur sonunda Flower bunu çağırır: ortak modeli 400 soruluk sınava sokar.
        tur=0 → hiç eğitim yapılmadan önceki durum."""
        model, tok = ortak.model_yukle(device)
        set_peft_model_state_dict(model, arrays.to_torch_state_dict())
        sonuc = ortak.alan_bazli_degerlendir(model, test, tok, device)
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

        # Ağ trafiği: her turda her istemciye LoRA gider (indirme) ve geri gelir (yükleme)
        ag_mb = tur * 2 * 2 * lora_mb  # tur × 2 istemci × 2 yön × LoRA boyutu
        ortak.sonuc_yazdir(f"\n[Sunucu] tur {tur}", sonuc)
        print(f"[Sunucu] şu ana kadar ağda giden toplam: {ag_mb:.1f} MB")

        kayitlar.append({
            "tur": tur,
            "dogruluk_tip": round(sonuc["dogruluk"]["tip"], 1),
            "dogruluk_finans": round(sonuc["dogruluk"]["finans"], 1),
            "dogruluk_ortalama": round(sonuc["dogruluk"]["ortalama"], 1),
            "loss_tip": round(sonuc["loss"]["tip"], 4),
            "loss_finans": round(sonuc["loss"]["finans"], 4),
            "toplam_ag_mb": round(ag_mb, 2),
            "gecen_dakika": round((time.time() - baslama) / 60, 2),
        })
        return MetricRecord({k: float(v) for k, v in kayitlar[-1].items()})

    # --- 2) FedAvg stratejisi ---
    # fraction_train=1.0     → her turda istemcilerin HEPSİ eğitime katılsın
    # fraction_evaluate=0.0  → istemcilerde ayrıca ölçüm yapma; sınavı sunucu yapıyor
    strateji = FedAvg(fraction_train=1.0, fraction_evaluate=0.0)
    strateji.start(
        grid=grid,
        initial_arrays=ArrayRecord(baslangic),
        num_rounds=tur_sayisi,
        evaluate_fn=sinav,
    )

    # --- 3) Sonuçları kaydet ---
    os.makedirs(config.SONUC_KLASORU, exist_ok=True)
    yol = os.path.join(config.SONUC_KLASORU, f"federe_{bolme}.csv")
    with open(yol, "w", newline="", encoding="utf-8") as f:
        yazici = csv.DictWriter(f, fieldnames=list(kayitlar[0].keys()))
        yazici.writeheader()
        yazici.writerows(kayitlar)
    print(f"\nSonuçlar kaydedildi: {yol}  (LoRA boyutu {lora_mb:.2f} MB)")
