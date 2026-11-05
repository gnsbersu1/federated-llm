# =====================================================================
#  Ortam testi: kütüphaneler kurulu mu, Qwen çalışıyor mu, LoRA kaç MB?
#  Çalıştırma:  python scripts/test_env.py
#  Windows dizüstünde (NVIDIA), Mac'te (MPS) ve CPU'da aynı şekilde çalışır.
# =====================================================================
import platform
import time

import datasets
import flwr
import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Kurulu sürümleri yazdır (bir hata olursa bana bunları göndermen işe yarar)
print("Python      :", platform.python_version())
print("PyTorch     :", torch.__version__)
print("transformers:", transformers.__version__)
print("peft        :", peft.__version__)
print("datasets    :", datasets.__version__)
print("flwr        :", flwr.__version__)


def cihaz_sec():
    """Hangi donanımda çalışacağımızı seçer: NVIDIA (cuda) > Apple (mps) > işlemci (cpu)."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")  # GPU gerçekten kullanılabiliyor mu? (eski sürücüde burada hata çıkar)
            return "cuda"
        except Exception as e:
            print("UYARI: NVIDIA kartı görünüyor ama kullanılamıyor:", str(e).splitlines()[0])
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = cihaz_sec()
print("Cihaz       :", device, f"({torch.cuda.get_device_name(0)})" if device == "cuda" else "")

# Tokenizer: metni sayılara (token'lara) çeviren parça. Model sadece sayı anlar.
tok = AutoTokenizer.from_pretrained(MODEL)
# Modeli indir (ilk seferde ~1 GB) ve seçilen cihaza taşı.
# GPU'da yarım hassasiyet (float16) daha az bellek kullanır; CPU'da float32 gerekir.
dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device)

# Sohbet formatında bir soru hazırla (chat template = modelin beklediği konuşma biçimi)
msgs = [{"role": "user", "content": "Merhaba! Kendini bir cümleyle tanıt."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp = tok(text, return_tensors="pt").to(device)

t = time.time()
out = model.generate(**inp, max_new_tokens=40, do_sample=False)
cevap = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
print("\nModelin cevabı:", cevap)
print(f"Süre: {time.time() - t:.1f} sn")

# LoRA ekle: modelin kendisi donar, sadece küçük ek matrisler eğitilir.
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, task_type="CAUSAL_LM"))
model.print_trainable_parameters()

# Federe öğrenmede ağda sadece bu eğitilebilir parametreler gidip gelecek
mb = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / 1e6
print(f"Ağda gidip gelecek LoRA boyutu: {mb:.2f} MB")
