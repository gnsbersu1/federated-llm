# =====================================================================
#  ADIM 3: FEDERE EĞİTİMİ BAŞLAT (Flower simülasyonu, bu bilgisayarda)
#  Çalıştırma:  python federe.py            (non-IID: A sadece tıp, B sadece finans)
#               python federe.py --iid      (IID: iki istemciye karışık veri)
#               python federe.py --tur 1    (sadece 1 tur, hızlı deneme)
#
#  Bu dosya sadece "flwr run" komutunu doğru ayarlarla çalıştırır ve
#  ekrandaki gereksiz mesajları (Ray'in Windows uyarıları) gizler.
#  Asıl kod: server_app.py (sunucu) ve client_app.py (istemci)
# =====================================================================
import os
import subprocess
import sys
from pathlib import Path

import config

bolme = "iid" if "--iid" in sys.argv else "noniid"
tur = sys.argv[sys.argv.index("--tur") + 1] if "--tur" in sys.argv else str(config.TUR_SAYISI)

# Ayarlar (ortam değişkenleri)
ortam = os.environ.copy()
ortam["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"  # Flower kütüphaneleri baştan indirmesin
ortam["FLWR_DISABLE_UPDATE_CHECK"] = "1"                      # "yeni sürüm var" uyarısı çıkmasın
ortam["PYTHONIOENCODING"] = "utf-8"                           # Türkçe karakterler bozulmasın
ortam["PYTHONUTF8"] = "1"
ortam["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
ortam["TRANSFORMERS_VERBOSITY"] = "error"

# flwr programı, şu an çalışan Python'un yanında duruyor
flwr = Path(sys.executable).parent / ("flwr.exe" if os.name == "nt" else "flwr")
komut = [str(flwr), "run", ".", "--stream", "--run-config", f"bolme='{bolme}' tur-sayisi={tur}"]
print("Çalıştırılıyor:", " ".join(komut), "\n")

# Çıktıyı satır satır oku; sadece işe yarayanları göster
gizle = ("(raylet)", "warnings.warn", "FutureWarning", "HF_TOKEN", "unauthenticated", "it/s]")
islem = subprocess.Popen(komut, env=ortam, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", cwd=os.path.dirname(__file__) or ".")
for satir in islem.stdout:
    satir = satir.rstrip()
    if not satir or any(g in satir for g in gizle) or satir.endswith(")\x1b[0m"):
        continue
    if "ClientAppActor" in satir and satir.split("]")[-1].strip() in ("", "\x1b[0m"):
        continue  # istemcilerden gelen boş satırlar
    print(satir, flush=True)
sys.exit(islem.wait())
