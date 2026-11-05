# =====================================================================
#  İSTEMCİ (ClientApp)  →  hastane (A) ya da finans şirketi (B)
#
#  Her turda Flower bu fonksiyonu çağırır:
#    1. Sunucudan gelen ortak LoRA ağırlıklarını al
#    2. SADECE kendi verinle birkaç adım eğit (veri bu bilgisayardan çıkmaz!)
#    3. Güncellenmiş LoRA ağırlıklarını ve örnek sayını sunucuya geri gönder
#
#  Baseline ile AYNI eğitim kodunu (ortak.egit) kullanır.
# =====================================================================
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from peft import get_peft_model_state_dict, set_peft_model_state_dict

import config
import ortak

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    # --- Ben kimim? ---
    # Flower her istemciye bir numara verir: 0 → İstemci A (hastane), 1 → İstemci B (finans)
    istemci = "AB"[context.node_config["partition-id"]]
    bolme = context.run_config["bolme"]          # "noniid" (gerçek senaryo) ya da "iid"
    tur = msg.content["config"]["server-round"]  # kaçıncı turdayız

    ortak.seed_ayarla(config.SEED + tur)  # her tur farklı ama tekrarlanabilir rastgelelik
    device = ortak.cihaz_sec()

    # --- 1) Modeli yükle ve sunucudan gelen ortak LoRA'yı içine koy ---
    model, tok = ortak.model_yukle(device)
    gelen_lora = msg.content["arrays"].to_torch_state_dict()
    set_peft_model_state_dict(model, gelen_lora)

    # --- 2) SADECE kendi verimle eğit ---
    veri = ortak.jsonl_oku(f"istemci{istemci}_{bolme}")
    yukleyici = ortak.veri_yukleyici(veri, tok, config.BATCH_BOYUTU, karistir=True)
    print(f"[İstemci {istemci}] tur {tur}: {len(veri)} örnekten {config.LOKAL_ADIM} adım eğitiliyor...")
    train_loss = ortak.egit(model, yukleyici, device, max_adim=config.LOKAL_ADIM)

    # --- 3) Sadece LoRA ağırlıklarını geri gönder (modelin kendisi ~1 GB, LoRA ~2 MB) ---
    lora = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
    cevap = RecordDict({
        "arrays": ArrayRecord(lora),
        "metrics": MetricRecord({
            "train_loss": train_loss,
            # FedAvg ağırlıklı ortalama için: kim daha çok veriyle eğittiyse sözü o kadar geçer
            "num-examples": config.LOKAL_ADIM * config.BATCH_BOYUTU * config.GRAD_BIRIKTIRME,
        }),
    })

    # Ekran kartı belleğini boşalt (sunucu da aynı kartı kullanıyor)
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    return Message(content=cevap, reply_to=msg)
