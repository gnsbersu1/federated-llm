# =====================================================================
#  ORTAK PARÇALAR
#  baseline.py ve (ileride) Flower istemcileri bu dosyadaki fonksiyonları
#  kullanır. Böylece merkezi ve federe eğitim AYNI kodla yapılır; tek fark
#  verinin nasıl dağıtıldığıdır. Bu, karşılaştırmanın adil olması için önemli.
# =====================================================================
import json
import os
import random

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


# ---------------------------------------------------------------------
# Donanım ve tekrarlanabilirlik
# ---------------------------------------------------------------------
def cihaz_sec():
    """NVIDIA (cuda) > Apple (mps) > işlemci (cpu) sırasıyla kullanılabilir olanı seçer."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_ayarla(seed=config.SEED):
    """Tüm rastgelelik kaynaklarını sabitler: aynı seed = aynı sonuç."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
def model_yukle(device, r=config.LORA_R):
    """Qwen'i yükler, üstüne LoRA ekler. Döndürür: (model, tokenizer)."""
    tok = AutoTokenizer.from_pretrained(config.MODEL_ADI)

    # Sayı hassasiyeti (dtype):
    #  - NVIDIA'da bfloat16: yarı bellek, hızlı ve eğitimde kararlı
    #  - Mac (MPS) ve CPU'da float32: daha yavaş ama güvenli (MPS'te float16 eğitimi kararsız olabiliyor)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_ADI, dtype=dtype).to(device)

    lora_ayari = LoraConfig(
        r=r,
        lora_alpha=2 * r,  # alpha = 2 × r kuralı (r=8 → 16)
        target_modules=config.LORA_HEDEF_KATMANLAR,
        lora_dropout=config.LORA_DROPOUT,
        task_type="CAUSAL_LM",
    )
    # get_peft_model: modelin orijinal ağırlıklarını DONDURUR, sadece LoRA eğitilebilir kalır.
    # (peft, LoRA ağırlıklarını otomatik olarak float32 tutar; küçük oldukları için sorun değil.)
    model = get_peft_model(model, lora_ayari)
    return model, tok


def lora_boyutu_mb(model):
    """Eğitilebilir (LoRA) parametrelerin kapladığı yer = ağda gidip gelecek miktar."""
    return sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / 1e6


# ---------------------------------------------------------------------
# Veri
# ---------------------------------------------------------------------
def jsonl_oku(ad):
    """data/<ad>.jsonl dosyasını okur, örnek listesi döndürür."""
    yol = os.path.join(config.VERI_KLASORU, f"{ad}.jsonl")
    with open(yol, encoding="utf-8") as f:
        return [json.loads(satir) for satir in f]


def tokenle(ornek, tok):
    """Bir konuşmayı token'lara çevirir ve 'labels' (öğrenilecek kısım) hazırlar.

    Önemli fikir: Model SADECE asistanın cevabından öğrensin, sorudan değil.
    Bu yüzden soru kısmının etiketini -100 yapıyoruz; -100 "burayı hesaba katma" demek.
    """
    tam_metin = tok.apply_chat_template(ornek["messages"], tokenize=False)
    # Sadece soru kısmı (+ "<|im_start|>assistant" başlangıcı): buraya kadarı öğrenilmeyecek
    soru_metni = tok.apply_chat_template(ornek["messages"][:1], tokenize=False, add_generation_prompt=True)

    input_ids = tok(tam_metin, add_special_tokens=False)["input_ids"][: config.MAX_TOKEN]
    soru_uzunlugu = len(tok(soru_metni, add_special_tokens=False)["input_ids"])

    labels = [-100] * soru_uzunlugu + input_ids[soru_uzunlugu:]
    return {"input_ids": input_ids, "labels": labels[: len(input_ids)]}


def veri_yukleyici(ornekler, tok, batch_boyutu, karistir):
    """Örnekleri batch'lere (paketlere) böler. Bir batch'teki örnekler aynı uzunluğa doldurulur."""
    tokenli = [tokenle(o, tok) for o in ornekler]

    def birlestir(batch):
        en_uzun = max(len(o["input_ids"]) for o in batch)
        ids, labels, maske = [], [], []
        for o in batch:
            eksik = en_uzun - len(o["input_ids"])
            ids.append(o["input_ids"] + [tok.pad_token_id] * eksik)  # kısa olanı doldur (padding)
            labels.append(o["labels"] + [-100] * eksik)               # doldurulan yerden öğrenme
            maske.append([1] * len(o["input_ids"]) + [0] * eksik)     # model doldurulanı görmezden gelsin
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(maske),
        }

    return DataLoader(tokenli, batch_size=batch_boyutu, shuffle=karistir, collate_fn=birlestir)


# ---------------------------------------------------------------------
# Eğitim ve ölçme
# ---------------------------------------------------------------------
def egit(model, yukleyici, device, epoch=1, max_adim=None, kayit=None):
    """Eğitim döngüsü. Her adımda: tahmin → loss → geri yayılım → LoRA'yı güncelle.

    Bir "adım" = GRAD_BIRIKTIRME kadar batch'in düzeltmesi toplanıp BİR KEZ uygulanması.
    kayit: her adımın loss'unu eklemek için bir liste (grafik çizmek için).
    Döndürür: ortalama eğitim loss'u.
    """
    model.train()  # eğitim modu (dropout açık)
    # AdamW: ağırlıkları loss'u azaltacak yönde güncelleyen yöntem (optimizer).
    # Sadece eğitilebilir (LoRA) parametreleri veriyoruz.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.OGRENME_ORANI
    )
    biriktir = config.GRAD_BIRIKTIRME
    toplam, adim, adim_loss = 0.0, 0, 0.0
    for ep in range(epoch):
        for i, batch in enumerate(yukleyici, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            cikti = model(**batch)                 # 1) tahmin + loss (labels verince model loss'u kendisi hesaplar)
            (cikti.loss / biriktir).backward()     # 2) her LoRA ağırlığının loss'a etkisini hesapla (gradyan), biriktir
            adim_loss += cikti.loss.item() / biriktir
            if i % biriktir != 0:
                continue                           # henüz yeterince batch birikmedi, sonrakine geç

            optimizer.step()                       # 3) biriken düzeltmeyle ağırlıkları biraz değiştir
            optimizer.zero_grad()                  # 4) bir sonraki adım için gradyanları sıfırla

            adim += 1
            toplam += adim_loss
            if kayit is not None:
                kayit.append({"adim": adim, "epoch": ep + 1, "train_loss": round(adim_loss, 4)})
            if adim % 25 == 0:
                print(f"  adım {adim:4d}  loss {adim_loss:.3f}  (ortalama {toplam / adim:.3f})")
            adim_loss = 0.0
            if max_adim and adim >= max_adim:
                return toplam / adim
    return toplam / max(adim, 1)


@torch.no_grad()  # ölçerken öğrenme yok: gradyan hesaplanmaz, daha hızlı ve az bellek
def degerlendir(model, yukleyici, device):
    """Test verisinde ortalama loss. Düşük = model doğru cevapları daha iyi tahmin ediyor."""
    model.eval()  # değerlendirme modu (dropout kapalı)
    toplam, n = 0.0, 0
    for batch in yukleyici:
        batch = {k: v.to(device) for k, v in batch.items()}
        toplam += model(**batch).loss.item()
        n += 1
    return toplam / max(n, 1)


@torch.no_grad()
def sik_tahmin_et(model, tok, ornek, device):
    """Modele şıklı bir soru sorar, hangi şıkkı seçtiğini döndürür (ör. 'C').

    Nasıl? Model cevabın İLK token'ı için her kelimeye bir olasılık verir.
    Sadece şık harflerine (A, B, C, D) bakıp en yüksek olasılıklı harfi seçiyoruz.
    Böylece model saçma bir şey yazsa bile her soruya net bir cevap almış oluruz.
    """
    model.eval()
    metin = tok.apply_chat_template(ornek["messages"][:1], tokenize=False, add_generation_prompt=True)
    girdi = tok(metin, return_tensors="pt").to(device)
    son_olasiliklar = model(**girdi).logits[0, -1]  # bir sonraki token için her kelimenin puanı
    harf_idleri = [tok.encode(h, add_special_tokens=False)[0] for h in ornek["harfler"]]
    return ornek["harfler"][int(son_olasiliklar[harf_idleri].argmax())]


def alan_bazli_degerlendir(model, test, tok, device):
    """Her alan için AYRI ölçer: doğruluk (% kaç soru doğru) ve test loss.

    Döndürür: {'dogruluk': {'tip': 41.5, 'finans': 78.0, 'ortalama': ...},
               'loss':     {'tip': 1.9,  'finans': 0.8,  'ortalama': ...}}

    Neden alan alan? Sadece tıp verisiyle eğitilen bir model tıpta iyi, finansta kötü olabilir.
    Federe öğrenmenin faydası tam olarak burada görünür: ortak model İKİ alanda da iyi mi?
    """
    sonuc = {"dogruluk": {}, "loss": {}}
    for alan in config.ALANLAR:
        alan_test = [o for o in test if o["category"] == alan]
        dogru = sum(sik_tahmin_et(model, tok, o, device) == o["dogru"] for o in alan_test)
        sonuc["dogruluk"][alan] = 100 * dogru / len(alan_test)
        yukleyici = veri_yukleyici(alan_test, tok, config.BATCH_BOYUTU, karistir=False)
        sonuc["loss"][alan] = degerlendir(model, yukleyici, device)
    for olcu in sonuc.values():
        olcu["ortalama"] = sum(olcu.values()) / len(olcu)
    return sonuc


def sonuc_yazdir(baslik, sonuc):
    """alan_bazli_degerlendir sonucunu okunaklı yazdırır."""
    dog = "  ".join(f"{k} %{v:.1f}" for k, v in sonuc["dogruluk"].items())
    los = "  ".join(f"{k} {v:.3f}" for k, v in sonuc["loss"].items())
    print(f"{baslik}  doğruluk: {dog}   |   loss: {los}")


@torch.no_grad()
def cevap_uret(model, tok, soru, device, max_token=80):
    """Modele bir soru sorar, cevabını metin olarak döndürür."""
    model.eval()
    metin = tok.apply_chat_template([{"role": "user", "content": soru}], tokenize=False, add_generation_prompt=True)
    girdi = tok(metin, return_tensors="pt").to(device)
    cikti = model.generate(**girdi, max_new_tokens=max_token, do_sample=False)  # do_sample=False: hep aynı cevap
    return tok.decode(cikti[0][girdi["input_ids"].shape[1]:], skip_special_tokens=True).strip()
