"""
Binance Futures sinyal botu (Telegram).
Her calistiginda: likit USDT perpetual coinleri tarar, kapanmis 1 saatlik mumda
sinyal veren coinleri siralar, en iyisini Telegram'a yollar. Acik sinyallerin
sonucunu (hedef / stop / sure doldu) takip eder. Emir VERMEZ, sadece sinyal yollar.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ------------------------------------------------------------------ AYARLAR
TOP_N = 60              # hacme gore en likit ilk N coin
MIN_HACIM = 30e6        # 24 saatlik hacim alt siniri (USDT)
RR = 2.0                # hedef = 2 x risk
MIN_RISK = 0.006        # stop mesafesi en az %0.6
MAX_RISK = 0.025        # stop mesafesi en fazla %2.5
GUNLUK_MAX = 4          # gunde (UTC) en fazla 4 sinyal
ACIK_MAX = 3            # ayni anda en fazla 3 acik sinyal
MAX_BAR = 24            # sinyal en fazla 24 saat gecerli
FEE_V = 0.0005 + 0.0002 # vadeli komisyon + kayma (her yon)
RISK_PCT = 1.0          # onerilen islem basi risk (% hesap)
MUM_SAYISI = 1000
DURUM_DOSYASI = "durum.json"
DISLA = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "AEUR", "EUR",
         "EURI", "USD1", "XUSD", "PAXG", "XAUT", "XAU", "XAG"}
# Hangi stratejiler acik? (gunici_test sonucuna gore degistirilecek)
STRATEJILER = os.environ.get("STRATEJILER",
                             "Trend pullback,Asya range kirilim,Dun tepe/dip + hacim").split(",")

TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

KAYNAKLAR = [
    ("futures", "https://fapi.binance.com/fapi/v1/ticker/24hr",
     "https://fapi.binance.com/fapi/v1/klines"),
    ("spot", "https://data-api.binance.vision/api/v3/ticker/24hr",
     "https://data-api.binance.vision/api/v3/klines"),
]


# ------------------------------------------------------------------ YARDIMCI
def getir(url, params=None, deneme=3):
    for _ in range(deneme):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(0.6)
    return None


def telegram(metin):
    if not TOKEN or not CHAT_ID:
        print("[Telegram ayari yok] ", metin)
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          data={"chat_id": CHAT_ID, "text": metin}, timeout=20)
        if r.status_code != 200:
            print("Telegram hatasi:", r.status_code, r.text[:200])
            return False
        return True
    except Exception as e:
        print("Telegram baglanti hatasi:", e)
        return False


def fmt(x):
    return f"{x:.6g}"


def durum_oku():
    if os.path.exists(DURUM_DOSYASI):
        try:
            with open(DURUM_DOSYASI, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"gun": "", "adet": 0, "acik": [], "gonderilen": [], "tamam": [], "hata_gun": ""}


def durum_yaz(d):
    d["gonderilen"] = d["gonderilen"][-300:]
    d["tamam"] = d["tamam"][-500:]
    with open(DURUM_DOSYASI, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ VERI
def kaynak_ve_liste():
    for ad, tick_url, kl_url in KAYNAKLAR:
        veri = getir(tick_url, deneme=2)
        if not veri or not isinstance(veri, list):
            print(f"Ulasilamadi: {ad}")
            continue
        liste = []
        for t in veri:
            s = t.get("symbol", "")
            if not s.endswith("USDT") or "_" in s or s[:-4] in DISLA:
                continue
            try:
                liste.append((s, float(t["quoteVolume"])))
            except Exception:
                continue
        liste = [x for x in liste if x[1] >= MIN_HACIM]
        liste.sort(key=lambda x: -x[1])
        return ad, kl_url, [s for s, _ in liste[:TOP_N]]
    return None, None, []


def mumlar(kl_url, sembol, bas_ms=None, limit=MUM_SAYISI):
    p = {"symbol": sembol, "interval": "1h", "limit": limit}
    if bas_ms:
        p["startTime"] = int(bas_ms)
    ham = getir(kl_url, p)
    if not ham:
        return None
    simdi = int(time.time() * 1000)
    # sadece kapanmis mumlar
    ham = [k for k in ham if int(k[0]) + 3600_000 <= simdi]
    if not ham:
        return None
    df = pd.DataFrame([[int(k[0]), float(k[1]), float(k[2]), float(k[3]),
                        float(k[4]), float(k[5])] for k in ham],
                      columns=["zaman", "o", "h", "l", "c", "v"])
    df.index = pd.to_datetime(df.pop("zaman"), unit="ms")
    return df


# ------------------------------------------------------------------ SINYAL
def atr_hesapla(h, l, c, n=14):
    onceki = c.shift()
    tr = pd.concat([h - l, (h - onceki).abs(), (l - onceki).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def hazirla(df):
    """gunici_test.py ile ayni mantik. Donus: {strateji: (long, short, sl_long, sl_short)}, skor"""
    o, h, l, c, v = df["o"], df["h"], df["l"], df["c"], df["v"]
    atr = atr_hesapla(h, l, c)
    ema20 = c.ewm(span=20, adjust=False).mean()

    u = c.resample("4h").last()
    u20 = u.ewm(span=20, adjust=False).mean()
    u50 = u.ewm(span=50, adjust=False).mean()
    idx4 = df.index.floor("4h")

    def al(seri):
        return pd.Series(seri.shift(1).reindex(idx4).values, index=df.index)

    c4, e20, e50 = al(u), al(u20), al(u50)
    up = (e20 > e50) & (c4 > e50)
    dn = (e20 < e50) & (c4 < e50)

    gun = df.index.floor("1D")
    saat = pd.Series(df.index.hour, index=df.index)

    def ilk_gun(sig):
        return sig & (sig.astype(int).groupby(gun).cumsum() == 1)

    t_l = l <= ema20 + 0.1 * atr
    t_s = h >= ema20 - 0.1 * atr
    s1_l = up & t_l.shift(1, fill_value=False) & (c > o) & (c > h.shift()) & (c > ema20)
    s1_s = dn & t_s.shift(1, fill_value=False) & (c < o) & (c < l.shift()) & (c < ema20)
    s1_sl_l = np.minimum(l, l.shift()) - 0.2 * atr
    s1_sl_s = np.maximum(h, h.shift()) + 0.2 * atr

    asya = saat < 7
    rng_h = h.where(asya).groupby(gun).transform("max")
    rng_l = l.where(asya).groupby(gun).transform("min")
    pencere = (saat >= 7) & (saat < 20)
    k_l = pencere & up & (c > rng_h) & (c.shift() <= rng_h)
    k_s = pencere & dn & (c < rng_l) & (c.shift() >= rng_l)
    ilk = ilk_gun(k_l | k_s)
    s2_l, s2_s = k_l & ilk, k_s & ilk

    pdh = pd.Series(h.resample("1D").max().shift(1).reindex(gun).values, index=df.index)
    pdl = pd.Series(l.resample("1D").min().shift(1).reindex(gun).values, index=df.index)
    vol_ok = v > 1.5 * v.rolling(24).mean().shift()
    q_l = up & (c > pdh) & (c.shift() <= pdh) & vol_ok
    q_s = dn & (c < pdl) & (c.shift() >= pdl) & vol_ok
    ilk3 = ilk_gun(q_l | q_s)
    s3_l, s3_s = q_l & ilk3, q_s & ilk3

    skor = v / v.rolling(24).mean().shift(1)
    return {
        "Trend pullback": (s1_l, s1_s, s1_sl_l, s1_sl_s),
        "Asya range kirilim": (s2_l, s2_s, rng_l, rng_h),
        "Dun tepe/dip + hacim": (s3_l, s3_s, c - 1.5 * atr, c + 1.5 * atr),
    }, skor


def son_mum_sinyalleri(df):
    """Son kapanmis mumda sinyal veren stratejiler: [(strateji, yon, sl, skor)]"""
    if len(df) < 400:
        return []
    sinyaller, skor = hazirla(df)
    sk = skor.iloc[-1]
    sk = 0.0 if pd.isna(sk) else float(sk)
    sonuc = []
    for ad in STRATEJILER:
        if ad not in sinyaller:
            continue
        sl_l, sg_s, sl_lv, sl_sv = sinyaller[ad]
        if bool(sl_l.iloc[-1]):
            sl = sl_lv.iloc[-1]
            if not pd.isna(sl):
                sonuc.append((ad, 1, float(sl), sk))
        elif bool(sg_s.iloc[-1]):
            sl = sl_sv.iloc[-1]
            if not pd.isna(sl):
                sonuc.append((ad, -1, float(sl), sk))
    return sonuc


# ------------------------------------------------------------------ ACIK SINYAL TAKIBI
def acik_takip(d, kl_url):
    kalan = []
    for p in d["acik"]:
        df = mumlar(kl_url, p["sembol"], bas_ms=p["bar_ms"], limit=MAX_BAR + 2)
        if df is None:
            kalan.append(p)
            continue
        yon, giris, sl, tp = p["yon"], p["giris"], p["sl"], p["tp"]
        risk = abs(giris - sl)
        sonuc, neden = None, ""
        for i, (_, r) in enumerate(df.iterrows()):
            if yon == 1:
                if r["l"] <= sl:
                    sonuc, neden = -1.0, "STOP"
                elif r["h"] >= tp:
                    sonuc, neden = RR, "HEDEF"
            else:
                if r["h"] >= sl:
                    sonuc, neden = -1.0, "STOP"
                elif r["l"] <= tp:
                    sonuc, neden = RR, "HEDEF"
            if sonuc is None and i + 1 >= MAX_BAR:
                sonuc, neden = yon * (r["c"] - giris) / risk, "SURE DOLDU"
            if sonuc is not None:
                break
        if sonuc is None:
            kalan.append(p)
            continue
        rp = risk / giris
        net = sonuc - 2 * FEE_V / rp
        d["tamam"].append({"sembol": p["sembol"], "strateji": p["strateji"], "yon": yon,
                           "brut_r": round(sonuc, 2), "net_r": round(net, 2),
                           "tarih": datetime.fromtimestamp(time.time(), timezone.utc).strftime("%Y-%m-%d %H:%M")})
        toplam = sum(x["net_r"] for x in d["tamam"])
        n = len(d["tamam"])
        kazanan = sum(1 for x in d["tamam"] if x["net_r"] > 0)
        simge = "✅" if net > 0 else "❌"
        telegram(f"{simge} SONUC: {p['sembol']} {'LONG' if yon == 1 else 'SHORT'} -> {neden}\n"
                 f"Brüt {sonuc:+.2f}R | Maliyet sonrası {net:+.2f}R\n"
                 f"Toplam ({n} sinyal): {toplam:+.1f}R, kazanan {kazanan}/{n}")
    d["acik"] = kalan


# ------------------------------------------------------------------ ANA AKIS
def main():
    if "--test" in sys.argv:
        ok = telegram("✅ Sinyal botu bağlantı testi başarılı. Bundan sonra sinyaller buraya gelecek.")
        print("Test mesaji gonderildi" if ok else "Test mesaji GONDERILEMEDI")
        sys.exit(0 if ok else 1)

    d = durum_oku()
    simdi = datetime.fromtimestamp(time.time(), timezone.utc)
    bugun = simdi.strftime("%Y-%m-%d")
    if d["gun"] != bugun:
        d["gun"], d["adet"] = bugun, 0

    kaynak, kl_url, semboller = kaynak_ve_liste()
    if not semboller:
        print("Veri kaynagina ulasilamadi")
        if d.get("hata_gun") != bugun:
            telegram("⚠️ Bot Binance verisine ulaşamadı. Sorun sürerse veri kaynağını değiştirmem gerekir, bana yaz.")
            d["hata_gun"] = bugun
        durum_yaz(d)
        sys.exit(0)
    print(f"Kaynak: {kaynak}, taranan coin: {len(semboller)}")

    acik_takip(d, kl_url)

    hak = min(GUNLUK_MAX - d["adet"], ACIK_MAX - len(d["acik"]))
    if hak > 0:
        mesgul = {p["sembol"] for p in d["acik"]}
        adaylar = []
        for s in semboller:
            if s in mesgul:
                continue
            df = mumlar(kl_url, s)
            if df is None:
                continue
            bar_ms = int(df.index[-1].timestamp() * 1000)
            if (simdi.timestamp() * 1000 - bar_ms) > 3 * 3600_000:
                continue  # bayat veri
            anahtar = f"{s}_{bar_ms}"
            if anahtar in d["gonderilen"]:
                continue
            kapanis = float(df["c"].iloc[-1])
            for ad, yon, sl, sk in son_mum_sinyalleri(df):
                rp = abs(kapanis - sl) / kapanis
                if not (MIN_RISK <= rp <= MAX_RISK):
                    continue
                adaylar.append((sk, s, ad, yon, sl, kapanis, rp, bar_ms, anahtar))
            time.sleep(0.05)
        adaylar.sort(key=lambda x: -x[0])
        print(f"Aday sayisi: {len(adaylar)}, hak: {hak}")
        kullanilan = set()
        for sk, s, ad, yon, sl, kapanis, rp, bar_ms, anahtar in adaylar:
            if hak <= 0:
                break
            if s in kullanilan:
                continue
            kullanilan.add(s)
            risk = abs(kapanis - sl)
            tp = kapanis + yon * RR * risk
            poz = RISK_PCT / 100 / rp * 100  # hesabin yuzde kaci kadar pozisyon
            yonad = "LONG (AL)" if yon == 1 else "SHORT (SAT)"
            simge = "🟢" if yon == 1 else "🔴"
            d["adet"] += 1
            metin = (f"{simge} {s}  {yonad}\n"
                     f"Strateji: {ad}\n"
                     f"Giriş: ~{fmt(kapanis)} (şu anki fiyat civarı)\n"
                     f"Stop: {fmt(sl)} (%{rp * 100:.2f} uzakta)\n"
                     f"Hedef: {fmt(tp)} ({RR:g}R)\n"
                     f"Risk %{RISK_PCT:g} ise pozisyon büyüklüğü ≈ hesabın %{poz:.0f}'i\n"
                     f"(100'ü aşıyorsa kaldıraç gerekir; küçük hesapta daha az risk al)\n"
                     f"Bugün: {d['adet']}/{GUNLUK_MAX} sinyal | Hacim skoru {sk:.1f}\n"
                     f"24 saat içinde sonuçlanmazsa kapat.\n"
                     f"Bu bir sinyaldir, emir bot tarafından verilmez. Yatırım tavsiyesi değildir.")
            if telegram(metin):
                d["gonderilen"].append(anahtar)
                d["acik"].append({"sembol": s, "strateji": ad, "yon": yon, "giris": kapanis,
                                  "sl": sl, "tp": tp, "bar_ms": bar_ms + 3600_000})
                hak -= 1
            else:
                d["adet"] -= 1
    else:
        print("Bugunluk limit dolu veya acik sinyal limiti dolu")

    durum_yaz(d)


if __name__ == "__main__":
    main()
