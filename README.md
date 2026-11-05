# Federe LoRA ile Qwen Fine-Tuning

Qwen2.5-0.5B-Instruct modelini **LoRA** ve **federe öğrenme** (Flower, FedAvg) ile eğitme deneyi.
Veri istemcide kalır; ağda sadece küçük LoRA ağırlıkları (~2 MB) gidip gelir, modelin tamamı (~1 GB) değil.

## Plan

1. Bu bilgisayarda (Windows + NVIDIA): veri hazırlama → baseline → Flower simülasyonu → deneyler
2. Lab'da (3 Mac mini): Mac 1 sunucu, Mac 2 İstemci A, Mac 3 İstemci B

## Klasörler

| Dosya / klasör | Ne işe yarar |
|---|---|
| `config.py` | Tüm ayarlar (model, LoRA, veri, eğitim) tek yerde |
| `scripts/test_env.py` | Ortam testi: kütüphaneler, Qwen cevabı, LoRA boyutu |
| `data/` | Hazırlanan veri (GitHub'a gitmez) |
| `results/` | Deney sonuçları (GitHub'a gitmez) |

## Bu bilgisayarda çalıştırma

```
C:\Users\bersu\fedlora\Scripts\activate
python scripts/test_env.py
```
