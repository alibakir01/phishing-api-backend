import json
import os
import threading
from datetime import datetime, timezone
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import tldextract
from scipy.sparse import csr_matrix, hstack
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from feature_extraction import extract_features


MODEL_PATH = "final_phishing_model_tfidf.pkl"


WHITELIST_DOMAINS = {
    # 🎓 EĞİTİM, AKADEMİ VE VERİ BİLİMİ (Senin favorilerin)
    "udemy.com", "coursera.org", "edx.org", "khanacademy.org", "datacamp.com", 
    "kaggle.com", "canvas.instructure.com", "blackboard.com", "quizlet.com", 
    "researchgate.net", "academia.edu", "agu.edu.tr", "moodle.org", "udacity.com",
    "pluralsight.com", "freecodecamp.org", "w3schools.com", "geeksforgeeks.org",

    # 💻 YAZILIM, BULUT VE TEKNOLOJİ DEVLERİ
    "github.com", "gitlab.com", "bitbucket.org", "stackoverflow.com", "docker.com",
    "microsoft.com", "google.com", "apple.com", "amazon.com", "aws.amazon.com",
    "azure.com", "cloudflare.com", "digitalocean.com", "heroku.com", "googleapis.com", "windows.net", "ibm.com",
    "oracle.com", "cisco.com", "intel.com", "amd.com", "nvidia.com", "atlassian.net",
    "postman.com", "figma.com", "notion.so", "slack.com", "trello.com",

    # 🇹🇷 TÜRKİYE'NİN EN ÇOK KULLANILAN VE GÜVENLİ SİTELERİ (E-Devlet, Alışveriş vb.)
    "turkiye.gov.tr", "e-devlet.gov.tr", "mhrs.gov.tr", "meb.gov.tr", "gib.gov.tr",
    "sahibinden.com", "trendyol.com", "hepsiburada.com", "n11.com", "ciceksepeti.com",
    "yemeksepeti.com", "getir.com", "migros.com.tr", "teknosa.com", "vatanbilgisayar.com",
    "mediamarkt.com.tr", "turkcell.com.tr", "vodafone.com.tr", "turktelekom.com.tr",
    "eksi-sozluk.com", "donanimhaber.com", "webtekno.com", "shiftdelete.net",
    "thy.com", "pegasus.com.tr", "obilet.com", "enuygun.com", "letgo.com",

    # 🏦 TÜRKİYE VE GLOBAL BANKALAR / FİNANS (En çok False Positive yiyenler)
    "garantibbva.com.tr", "isbank.com.tr", "ziraatbank.com.tr", "ykb.com", "akbank.com",
    "vakifbank.com.tr", "halkbank.com.tr", "enpara.com", "papara.com", "qnbfinansbank.com",
    "teb.com.tr", "denizbank.com", "kuveytturk.com.tr", "paypal.com", "stripe.com",
    "wise.com", "payoneer.com", "mastercard.com", "visa.com", "binance.com",

    # 📱 SOSYAL MEDYA, İLETİŞİM VE İÇERİK
    "linkedin.com", "youtube.com", "twitter.com", "x.com", "facebook.com", 
    "instagram.com", "tiktok.com", "reddit.com", "twitch.tv", "discord.com", 
    "whatsapp.com", "telegram.org", "zoom.us", "skype.com", "pinterest.com",
    "medium.com", "wordpress.com", "tumblr.com", "vimeo.com", "spotify.com",

    # 📰 HABER, BİLGİ VE ANSİKLOPEDİ
    "wikipedia.org", "bbc.com", "cnn.com", "nytimes.com", "reuters.com",
    "bloomberg.com", "forbes.com", "theguardian.com", "wsj.com", "wired.com",

    # 🛠️ KURUMSAL BULUT VE MAİL SERVİSLERİ
    "sharepoint.com", "onedrive.live.com", "1drv.ms", "dropbox.com", "box.com",
    "wefer.com", "outlook.com", "yahoo.com", "protonmail.com", "icloud.com",
    "mail.ru", "yandex.com", "yandex.com.tr", "zoho.com", "salesforce.com",

    # 🌐 GLOBAL E-TİCARET VE EĞLENCE
    "ebay.com", "aliexpress.com", "alibaba.com", "walmart.com", "target.com",
    "netflix.com", "disneyplus.com", "primevideo.com", "hulu.com", "hbo.com",
    "steampowered.com", "epicgames.com", "ea.com", "ubisoft.com", "playstation.com"

    # 🛒 AMAZON EKOSİSTEMİ VE ÜLKE UZANTILARI
    "amazon.com.tr", "amazon.de", "amazon.co.uk", "amazon.fr", "amazon.it", 
    "amazon.es", "amazon.ca", "amazon.co.jp", "amazon.com.au", "amazon.nl",
    "imdb.com", "goodreads.com", "audible.com", "zappos.com", "w Woot.com", 
    "shopbop.com", "aws.amazon.com", "aws.dev", "amazon.jobs",
}

class URLRequest(BaseModel):
    url: str


class PredictionResponse(BaseModel):
    is_phishing: bool
    confidence_score: float  # yüzde olarak (0-100)
    reasons: List[str]


class FeedbackRequest(BaseModel):
    """Human-in-the-loop feedback for false positive/negative review (offline only — no model training)."""

    url: str
    confidence_score: float
    predicted_as_phishing: bool
    user_comments: Optional[str] = None


FEEDBACK_LOG_PATH = "feedback_logs.json"
_feedback_log_lock = threading.Lock()


def _append_feedback_entry(entry: dict) -> None:
    """Thread-safe append to a JSON array file (manual review pipeline; does not touch the model)."""
    with _feedback_log_lock:
        logs: List[dict] = []
        if os.path.isfile(FEEDBACK_LOG_PATH):
            try:
                with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                    if raw:
                        data = json.loads(raw)
                        if isinstance(data, list):
                            logs = data
            except (json.JSONDecodeError, OSError):
                logs = []
        logs.append(entry)
        with open(FEEDBACK_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)


def load_model(path: str = MODEL_PATH):
    bundle = joblib.load(path)
    model = bundle["model"]
    selector = bundle["selector"]
    preprocessors = bundle["preprocessors"]
    return model, selector, preprocessors


model, selector, preprocessors = load_model()

app = FastAPI(title="Phishing URL Detection API")

# CORS ayarları (frontend için serbest)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_feature_row(url: str) -> pd.DataFrame:
    """Tek bir URL için v2 feature pipeline'ını çalıştır ve tek satırlık DataFrame döndür."""
    df = pd.DataFrame([{"url": url, "label": 0}])
    df_feat = extract_features(df.copy(), url_col="url")
    return df_feat


def get_registered_domain(url: str) -> str:
    """Extract registered domain + suffix using tldextract (e.g., google.com)."""
    ext = tldextract.extract(url)
    if not ext.domain or not ext.suffix:
        return ""
    return f"{ext.domain}.{ext.suffix}".lower()


def build_feature_vector(df_feat: pd.DataFrame) -> np.ndarray:
    """
    TF-IDF + sayısal özellikleri birleştirerek, SelectKBest ile daraltılmış
    model girişi üret.
    """
    # Numeric features
    X_num_df = df_feat.drop(columns=["label"])
    non_numeric = X_num_df.select_dtypes(include=["object", "string"]).columns.tolist()
    if non_numeric:
        X_num_df = X_num_df.drop(columns=non_numeric)

    num_feature_names = preprocessors["num_feature_names"]
    for col in num_feature_names:
        if col not in X_num_df.columns:
            X_num_df[col] = 0.0
    X_num_df = X_num_df[num_feature_names]

    X_num = X_num_df.values.astype(np.float32)
    num_scaler = preprocessors["num_scaler"]
    X_num_scaled = num_scaler.transform(X_num)
    X_num_sparse = csr_matrix(X_num_scaled)

    # TF-IDF domain + path
    url_clean = str(df_feat["url_clean"].iloc[0])
    ext = tldextract.extract(url_clean)
    domain_text = [f"{ext.domain}.{ext.suffix}".lower() if ext.domain and ext.suffix else ""]

    from urllib.parse import urlparse

    def get_path_text_single(u: str) -> str:
        u2 = (u or "").strip()
        if not u2:
            return ""
        if not u2.lower().startswith(("http://", "https://")):
            u2 = "http://" + u2
        try:
            p = urlparse(u2)
        except ValueError:
            return ""
        path = p.path or ""
        query = p.query or ""
        if query:
            return f"{path}?{query}"
        return path

    path_text = [get_path_text_single(url_clean)]

    domain_vec = preprocessors["domain_vec"]
    path_vec = preprocessors["path_vec"]
    X_domain = domain_vec.transform(domain_text)
    X_path = path_vec.transform(path_text)

    X_all = hstack([X_num_sparse, X_domain, X_path], format="csr")
    X_sel = selector.transform(X_all)
    return X_sel


def explain_risk(df_feat: pd.DataFrame) -> List[str]:
    """URL neden riskli olabilir? Basit, açıklayıcı ipuçları üret."""
    row = df_feat.iloc[0]
    reasons: List[str] = []

    # Shannon entropy
    entropy = row.get("shannon_entropy", 0.0)
    if entropy >= 4.0:
        reasons.append("URL karakterlerinin entropisi yüksek (rastgele karakter dizileri).")

    # Dangerous TLD
    if int(row.get("is_dangerous_tld", 0)) == 1:
        reasons.append("Yüksek riskli TLD (uzantı) tespit edildi.")

    # IP address
    if int(row.get("has_ip", 0)) == 1:
        reasons.append("Alan adı yerine IP adresi kullanılıyor.")

    # Brand spoofing
    if int(row.get("has_brand_spoof", 0)) == 1:
        reasons.append("URL içinde popüler marka ismi şüpheli şekilde kullanılmış (brand spoofing).")

    # Repeated tokens
    if int(row.get("has_repeated_token", 0)) == 1:
        reasons.append("Tekrarlayan anahtar kelime/tanım parçaları tespit edildi (ör. login-login).")

    # Digit-letter ratio
    dl_ratio = row.get("digit_letter_ratio", 0.0)
    if dl_ratio >= 0.6:
        reasons.append("URL'de harflere göre çok sayıda rakam bulunuyor.")

    # Suspicious keywords
    suspicious_flags = [
        "kw_login",
        "kw_bank",
        "kw_account",
        "kw_update",
        "kw_free",
        "kw_lucky",
        "kw_service",
    ]
    if any(int(row.get(col, 0)) == 1 for col in suspicious_flags):
        reasons.append("URL içinde şüpheli anahtar kelimeler (login, bank, account vb.) tespit edildi.")

    # Çok uzun URL
    url_len = row.get("url_length_clean", 0)
    if url_len and url_len > 120:
        reasons.append("URL olağandışı derecede uzun.")

    # 1-2 kısa ipucuyla sınırlamak için gerekirse kırp
    if len(reasons) > 2:
        return reasons[:2]
    return reasons


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """
    Append user feedback to feedback_logs.json for offline review by data scientists.
    Does NOT retrain or modify the XGBoost model.
    """
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "url": request.url,
            "confidence_score": float(request.confidence_score),
            "predicted_as_phishing": bool(request.predicted_as_phishing),
            "user_comments": (request.user_comments.strip() if request.user_comments else None),
        }
        _append_feedback_entry(entry)
        return {
            "status": "ok",
            "message": "Feedback logged for review.",
        }
    except Exception as e:
        print("[ERROR] /feedback exception:", repr(e))
        raise HTTPException(status_code=500, detail="Could not persist feedback.")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: URLRequest):
    """
    Tek URL için tahmin endpoint'i. (Açık Yönlendirme Korumalı ve Dengeli Heuristic Sürüm)
    """
    try:
        url_lower = request.url.lower()

        # 0. YEREL GELİŞTİRİCİ ORTAMI KONTROLÜ (Localhost Bypass)
        if "localhost" in url_lower or "127.0.0.1" in url_lower:
            return PredictionResponse(
                is_phishing=False,
                confidence_score=0.01,
                reasons=["Yerel Geliştirici Ortamı (Localhost / 127.0.0.1) tespit edildi."],
            )

        # 1. KESİN WHITELIST KONTROLÜ (Open Redirect - Açık Yönlendirme Korumalı!)
        ext = tldextract.extract(request.url)
        reg_domain = f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else ""
        
        # Eğer linkin içinde =http, redirect=, url= varsa bu bir sektirme taktiğidir! VIP'yi iptal et.
        is_open_redirect = "=http" in url_lower or "redirect" in url_lower or "url=" in url_lower or "?q=http" in url_lower

        if reg_domain in WHITELIST_DOMAINS and not is_open_redirect:
            return PredictionResponse(
                is_phishing=False,
                confidence_score=0.01,
                reasons=[f"Güvenilir Altyapı / Whitelist ({reg_domain})"],
            )

        df_feat = build_feature_row(request.url)
        X_sel = build_feature_vector(df_feat)

        probs = model.predict_proba(X_sel)[0]
        prob_safe = float(probs[0])
        prob_phish = float(probs[1])

        # --- KRİTİK SİNYAL KURALLARI ---
        row = df_feat.iloc[0]
        is_whitelisted = reg_domain in WHITELIST_DOMAINS
        
        # tldextract ile suffix'in tr ile bitip bitmediğini yakala (.com.tr, .edu.tr vb.)
        is_tr_domain = ext.suffix and ext.suffix.endswith("tr")

        has_ip = bool(row.get("has_ip", 0))
        has_at = bool(row.get("has_at", 0))
        has_brand_spoof = bool(row.get("has_brand_spoof", 0)) 
        
        suspicious_cols = ["kw_login", "kw_bank", "kw_account", "kw_update", "kw_free", "kw_lucky", "kw_service"]
        has_dangerous_word = any(bool(row.get(col, 0)) for col in suspicious_cols)

        # 1. Marka Taklidi (Typosquatting): Hacker'a acımak yok!
        if has_brand_spoof and not is_whitelisted:
            prob_phish = max(prob_phish, 0.95)

        # 2. IP veya @ varsa (Obfuscation): Hala çok şüpheli ama %85 ile sınırla.
        if (has_ip or has_at) and not is_whitelisted:
            prob_phish = max(prob_phish, 0.85)

        # 3. Tehlikeli Kelime Cezası: +%15 ceza (Dengeli oran)
        if has_dangerous_word and not is_whitelisted:
            prob_phish = min(1.0, prob_phish + 0.15)

        # 4. TÜRKİYE İNDİRİMİ (Blue Team Koruması): 
        # Eğer site .tr uzantılıysa, içinde IP veya marka taklidi yoksa modelin paranoyasını -%30 azalt.
        if is_tr_domain and not has_ip and not has_brand_spoof:
            prob_phish = max(0.01, prob_phish - 0.30)
       
        # 5. SEO VE İÇERİK (HABER/BLOG/TARİF) TOLERANSI
        # Eğer linkte IP yoksa, marka taklidi yoksa ve "login, bank, password" gibi tehlikeli kelimeler HİÇ YOKSA:
        if not has_ip and not has_brand_spoof and not has_dangerous_word and not is_whitelisted:
            # Sırf sonunda "4213449" gibi id'ler veya tireler var diye modelin çıldırmasını engelle.
            if prob_phish > 0.45:
                prob_phish = 0.45    

        # Skorları yeniden dengele
        prob_safe = 1.0 - prob_phish

        # Çok seviyeli threshold
        threshold_suspicious = 0.40   # Şüpheli barajı
        threshold_phishing = 0.70     # Kesin phishing

        is_phishing = prob_phish >= threshold_suspicious
        confidence = prob_phish if is_phishing else prob_safe
        confidence_percent = float(confidence * 100.0)

        reasons = explain_risk(df_feat) if is_phishing else []

        # Türkçe Siber İstihbarat Kalkanı
        turkish_threat_words = [
            "odeme", "kart", "sifre", "giris", "dogrulama", 
            "fatura", "hesap", "guvenlik", "iade", "kargo", 
            "aidat", "kredi", "onay", "musteri", "sube", "guncelleme"
        ]
        
        turkish_safe_context = [
            "konu", "sosyal", "forum", "thread", "blog", "haber", 
            "checkout", "sepet", "urun", "kategori", "magaza", "iletisim", "product"
        ]
        
        if not is_whitelisted:
            has_safe_context = any(safe_word in url_lower for safe_word in turkish_safe_context)
            is_http = url_lower.startswith("http://") # SSL Sertifikası yok mu?
            
            for word in turkish_threat_words:
                if word in url_lower:
                    # E-ticaret veya forumsa es geç
                    if has_safe_context:
                        reasons.append(f"Ignored Turkish threat '{word}' due to safe context (forum/e-commerce).")
                        break
                        
                    # 1. SENARYO: Site HTTP (Güvensiz) ve tehlikeli kelime var. ACIMAK YOK!
                    if is_http:
                        confidence_percent = max(95.0, confidence_percent + 45.0)
                        is_phishing = True
                        reasons.append(f"CRITICAL: Unencrypted HTTP connection asking for sensitive data ('{word}').")
                        break
                        
                    # 2. SENARYO: HTTPS ama uzantı .tr değil.
                    elif not is_tr_domain:
                        confidence_percent = max(65.0, min(95.0, confidence_percent + 25.0))
                        is_phishing = True if confidence_percent >= 40.0 else False
                        reasons.append(f"Turkish threat keyword on non-TR domain: '{word}'")
                        break
                        
                    # 3. SENARYO: HTTPS ve .tr uzantısı var ama XGBoost zaten şüphelenmiş
                    elif confidence_percent > 20.0:
                        confidence_percent = max(55.0, min(85.0, confidence_percent + 15.0))
                        is_phishing = True if confidence_percent >= 40.0 else False
                        reasons.append(f"Suspicious Turkish keyword detected: '{word}'")
                        break
        
        dev_file_extensions = [".dart", ".py", ".js", ".json", ".css", ".md", ".txt"]
        
        # URL'de parametre (?) varsa at, sadece uzantıya bak
        url_path_only = url_lower.split("?")[0]
        is_dev_file = any(url_path_only.endswith(ext) for ext in dev_file_extensions)
        is_github_raw = "raw.githubusercontent.com" in url_lower

        if is_dev_file or is_github_raw:
            is_phishing = False
            confidence_percent = 5.0  # Skoru acımadan %5'e çakıyoruz
            reasons = ["Ignored risk: URL points to a raw developer file or source code."]
        

        # Log dosyasına yaz
        if is_phishing:
            try:
                from datetime import datetime
                line = (
                    f"{datetime.utcnow().isoformat()}Z\t"
                    f"url={request.url}\tprob_safe={prob_safe:.4f}\tprob_phish={prob_phish:.4f}\n"
                )
                with open("potential_false_positives.log", "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception:
                pass

        return PredictionResponse(
            is_phishing=bool(is_phishing),
            confidence_score=round(confidence_percent, 2),
            reasons=reasons,
        )

    except Exception as e:
        print("[ERROR] /predict exception:", repr(e))
        # DİKKAT: Hata durumunda Güvenli DEME! Şüpheli kabul et veya hata kodu dön.
        return PredictionResponse(
            is_phishing=True, # Fail-Secure (Hata varsa risklidir!)
            confidence_score=99.9,
            reasons=[f"Sistem analizi başarısız (Şüpheli URL yapısı)"],
        )
# Örnek kullanım:
# uvicorn api:app --host 0.0.0.0 --port 8000 --reload

