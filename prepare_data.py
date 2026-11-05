# =====================================================================
#  ADIM 1: Veri hazırlama (TIP + FİNANS, şıklı sorular)
#  Çalıştırma:  python prepare_data.py
#
#  Senaryo: İstemci A bir hastane, İstemci B bir finans şirketi.
#  Her biri sadece kendi alanının verisine sahip (non-IID).
#  Karşılaştırma için bir de "karışık" (IID) bölme yapıyoruz.
#
#  Her örnek ŞIKLI bir soru, doğru cevap belli. Böylece eğitimden sonra
#  "model tıp sorularının yüzde kaçını doğru biliyor?" diye ölçebiliriz.
#
#  Kaydedilen dosyalar (data/):
#    test.jsonl              200 tıp + 200 finans sınav sorusu (eğitimde HİÇ kullanılmaz)
#    istemciA_noniid.jsonl   1000 tıp sorusu      (hastane)
#    istemciB_noniid.jsonl   1000 finans sorusu   (finans şirketi)
#    istemciA_iid.jsonl      aynı 2000 sorunun karışık yarısı
#    istemciB_iid.jsonl      diğer yarısı
# =====================================================================
import json
import os
import random
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

import config

random.seed(config.SEED)  # sabit seed: her çalıştırmada aynı bölme
tok = AutoTokenizer.from_pretrained(config.MODEL_ADI)
HARFLER = "ABCDE"


def sikli_ornek(soru, secenekler, dogru_index, alan):
    """Bir soruyu şıklarıyla birlikte sohbet formatına çevirir.

    Kullanıcı:  <soru>  A. ...  B. ...  C. ...  Answer with the letter only.
    Asistan:    C. <doğru şıkkın metni>
    """
    harfler = HARFLER[: len(secenekler)]
    sik_metni = "\n".join(f"{h}. {s}" for h, s in zip(harfler, secenekler))
    dogru = harfler[dogru_index]
    return {
        "messages": [
            {"role": "user", "content": f"{soru}\n\n{sik_metni}\n\nAnswer with the letter only."},
            {"role": "assistant", "content": f"{dogru}. {secenekler[dogru_index]}"},
        ],
        "category": alan,
        "harfler": harfler,   # bu soruda hangi şıklar var (ör. "ABCD")
        "dogru": dogru,       # doğru şıkkın harfi (ölçerken kullanacağız)
    }


# ---------------------------------------------------------------------
# Tıp: MedMCQA
# ---------------------------------------------------------------------
def tip_hazirla():
    """Eğitim soruları 'train' bölümünden, sınav soruları ayrı 'validation' bölümünden alınır."""
    def cevir(satir):
        return sikli_ornek(
            satir["question"],
            [satir["opa"], satir["opb"], satir["opc"], satir["opd"]],
            int(satir["cop"]),  # cop = correct option: 0=A, 1=B, 2=C, 3=D
            "tip",
        )

    def uygun(satir):
        # sadece tek doğru cevaplı sorular, şıkları boş olmayanlar
        return satir["choice_type"] == "single" and all(satir[k] for k in ("opa", "opb", "opc", "opd"))

    egitim = [cevir(s) for s in load_dataset(config.ALANLAR["tip"], split="train") if uygun(s)]
    test = [cevir(s) for s in load_dataset(config.ALANLAR["tip"], split="validation") if uygun(s)]
    return test, egitim


# ---------------------------------------------------------------------
# Finans: FinGPT duygu analizi (haber olumsuz mu, nötr mü, olumlu mu?)
# ---------------------------------------------------------------------
def finans_hazirla():
    """Bu veri setinin ayrı bir test bölümü yok; karıştırıp kendimiz ayıracağız."""
    etiketler = ["negative", "neutral", "positive"]  # A, B, C
    ornekler, gorulen = [], set()
    for satir in load_dataset(config.ALANLAR["finans"], split="train"):
        # Bu veri setinde aynı haber birden fazla kez geçiyor. Tekrarları atıyoruz;
        # yoksa aynı haber hem eğitimde hem sınavda çıkabilir (= kopya çekmek).
        if satir["input"] in gorulen:
            continue
        gorulen.add(satir["input"])
        cikti = satir["output"].lower()
        # bazı cevaplar "moderately positive" gibi; içinde geçen kelimeye göre etiketle
        etiket = next((e for e in etiketler if e in cikti), None)
        if etiket is None:
            continue
        ornekler.append(sikli_ornek(
            f"What is the sentiment of this financial news?\n\n{satir['input']}",
            etiketler, etiketler.index(etiket), "finans"))
    random.shuffle(ornekler)
    yarim = len(ornekler) // 2
    return ornekler[:yarim], ornekler[yarim:]  # ilk yarıdan test, ikinci yarıdan eğitim seçilecek


# ---------------------------------------------------------------------
# Her alandan gereken sayıda soru seç (çok uzunları atla)
# ---------------------------------------------------------------------
def sec(ornekler, adet):
    random.shuffle(ornekler)
    secilen = []
    for o in ornekler:
        metin = tok.apply_chat_template(o["messages"], tokenize=False)
        if len(tok(metin)["input_ids"]) <= config.MAX_TOKEN:
            secilen.append(o)
        if len(secilen) == adet:
            return secilen
    raise SystemExit("Yeterli örnek yok! ALAN_BASINA_EGITIM / ALAN_BASINA_TEST değerini küçült.")


test, egitim = [], {}
for alan, hazirla in [("tip", tip_hazirla), ("finans", finans_hazirla)]:
    print(f"[{alan}] {config.ALANLAR[alan]} indiriliyor...")
    alan_test, alan_egitim = hazirla()
    alan_test = sec(alan_test, config.ALAN_BASINA_TEST)
    # Güvenlik kontrolü: sınavdaki bir soru eğitim havuzunda da varsa onu eğitimden çıkar
    sinav_sorulari = {o["messages"][0]["content"] for o in alan_test}
    alan_egitim = [o for o in alan_egitim if o["messages"][0]["content"] not in sinav_sorulari]
    test += alan_test
    egitim[alan] = sec(alan_egitim, config.ALAN_BASINA_EGITIM)

# ---------------------------------------------------------------------
# Non-IID: her istemci SADECE kendi alanını görür (A = hastane, B = finans şirketi)
# ---------------------------------------------------------------------
istemciA_noniid = egitim[config.NONIID_ISTEMCI_ALANI["A"]]
istemciB_noniid = egitim[config.NONIID_ISTEMCI_ALANI["B"]]

# ---------------------------------------------------------------------
# IID: AYNI 2000 soruyu karıştırıp ikiye böl → her istemciye yarı tıp, yarı finans
# (Aynı sorular, sadece paylaştırma farklı: aradaki fark sadece dağılımdan gelir.)
# ---------------------------------------------------------------------
hepsi = istemciA_noniid + istemciB_noniid
random.shuffle(hepsi)
yarim = len(hepsi) // 2
istemciA_iid, istemciB_iid = hepsi[:yarim], hepsi[yarim:]


# ---------------------------------------------------------------------
# Kaydet
# ---------------------------------------------------------------------
def kaydet(veri, ad):
    """Her satırı bir örnek olan .jsonl dosyası yazar (metin editöründe açıp okuyabilirsin)."""
    yol = os.path.join(config.VERI_KLASORU, f"{ad}.jsonl")
    with open(yol, "w", encoding="utf-8") as f:
        for o in veri:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"  {yol:32s} {len(veri):5d} örnek")


os.makedirs(config.VERI_KLASORU, exist_ok=True)
print("\nKaydedilen dosyalar:")
kaydet(test, "test")
kaydet(istemciA_iid, "istemciA_iid")
kaydet(istemciB_iid, "istemciB_iid")
kaydet(istemciA_noniid, "istemciA_noniid")
kaydet(istemciB_noniid, "istemciB_noniid")

parcalar = {"A_iid": istemciA_iid, "B_iid": istemciB_iid,
            "A_noniid": istemciA_noniid, "B_noniid": istemciB_noniid, "test": test}
print("\nAlan dağılımı (örnek sayısı):")
print(f"{'alan':10s}" + "".join(f"{ad:>10s}" for ad in parcalar))
for alan in config.ALANLAR:
    print(f"{alan:10s}" + "".join(f"{Counter(o['category'] for o in p)[alan]:10d}" for p in parcalar.values()))

print("\nDoğru cevap harflerinin dağılımı (test):")
for alan in config.ALANLAR:
    print(f"  {alan:8s}", dict(sorted(Counter(o["dogru"] for o in test if o["category"] == alan).items())))
