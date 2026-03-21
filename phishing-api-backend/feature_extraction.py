import os
import re
import math
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import pandas as pd


SHORTENER_DOMAINS = {
    "bit.ly",
    "t.co",
    "tinyurl.com",
    "goo.gl",
    "ow.ly",
    "buff.ly",
    "bitly.com",
    "is.gd",
    "adf.ly",
    "bit.do",
    "t.ly",
    "cutt.ly",
}


IPV4_REGEX = re.compile(
    r"""
    \b
    (?:25[0-5]|2[0-4]\d|1?\d{1,2})
    (?:\.
        (?:25[0-5]|2[0-4]\d|1?\d{1,2})
    ){3}
    \b
    """,
    re.VERBOSE,
)

# Simplified IPv6 presence check (we only care about existence, not validity)
IPV6_REGEX = re.compile(r"\b(?:[0-9a-fA-F]{0,4}:){2,}[0-9a-fA-F]{0,4}\b")

SUSPICIOUS_KEYWORDS = [
    "login", "bank", "account", "update", "free", "lucky", "service", 
    "verify", "password", "secure", "billing", "credential", "wallet"
]

BRAND_NAMES = [
    "google", "facebook", "apple", "microsoft", "paypal", "amazon", 
    "netflix", "instagram", "linkedin", "twitter", "binance", "ziraat", "garanti"
]


DANGEROUS_TLDS = {
    "xyz",
    "top",
    "club",
    "biz",
    "work",
    "info",
    "click",
    "link",
    "loan",
    "men",
    "party",
    "date",
    "kim",
    "download",
}

TRUSTED_TR_SUFFIXES = {
    "com.tr",
    "gov.tr",
    "edu.tr",
    "bel.tr",
    "org.tr",
    "net.tr",
    "k12.tr",
    "pol.tr",
    "mil.tr",
    "av.tr",
}

TRACKING_PARAMS = {
    "gclid",
    "gad_source",
    "gad_campaignid",
    "cid",
    "fbclid",
}


def strip_tracking_params(raw_url: str) -> str:
    """
    Remove marketing/tracking query params like gclid, utm_*, fbclid, etc.
    Returns sanitized URL string (scheme+netloc+path+filtered query+fragment).
    """
    if not raw_url:
        return raw_url
    try:
        parsed = urlparse(raw_url)
    except ValueError:
        return raw_url

    qs = parse_qsl(parsed.query, keep_blank_values=True)
    filtered = []
    for key, value in qs:
        lk = key.lower()
        if lk in TRACKING_PARAMS:
            continue
        if lk.startswith("utm_"):
            continue
        filtered.append((key, value))

    new_query = urlencode(filtered, doseq=True)
    sanitized = parsed._replace(query=new_query)
    return urlunparse(sanitized)


def normalize_url(url: str) -> str:
    """
    Return a 'clean' URL string without protocol, leading www, and surrounding whitespace.
    All downstream structural features should use this cleaned version so that:
    - https://google.com
    - http://google.com
    - www.google.com
    - google.com
    hepsi aynı temsile dönüşür.
    """
    if not isinstance(url, str):
        url = str(url)
    url = url.strip().strip('"').strip("'")
    # Remove trailing commas/spaces that might come from viewing artifacts
    url = url.rstrip(" ,")

    # First strip tracking/marketing parameters (gclid, utm_*, fbclid, etc.)
    url = strip_tracking_params(url)

    lower = url.lower()
    if lower.startswith("http://"):
        url = url[7:]
        lower = url.lower()
    elif lower.startswith("https://"):
        url = url[8:]
        lower = url.lower()

    # Remove leading slashes that might remain after protocol stripping
    while url.startswith("/"):
        url = url[1:]

    # Strip leading www.
    if lower.startswith("www."):
        url = url[4:]

    return url


def get_protocol_flags(url: str):
    """Return (has_https, has_http, no_protocol) flags based on raw URL."""
    if not isinstance(url, str):
        url = str(url)
    url_strip = url.strip()
    lower = url_strip.lower()
    has_https = int(lower.startswith("https://"))
    has_http = int(lower.startswith("http://") and not lower.startswith("https://"))
    no_protocol = int(not lower.startswith("http://") and not lower.startswith("https://"))
    return has_https, has_http, no_protocol


def parse_domain(url_raw: str, url_clean: str) -> str:
    """
    Extract domain from URL using urlparse.
    If there is no scheme, add a dummy http:// for parsing.
    """
    candidate = (url_raw or "").strip()
    lower = candidate.lower()
    def safe_parse(u: str):
        try:
            return urlparse(u)
        except ValueError:
            return None

    if not (lower.startswith("http://") or lower.startswith("https://")):
        candidate = "http://" + candidate.lstrip("/ ")

    parsed = safe_parse(candidate)
    netloc = parsed.netloc if parsed else ""

    # Some URLs may be like 'example.com/path' and end up in path instead
    if not netloc:
        parsed2 = safe_parse("http://" + (url_clean or "").lstrip("/ "))
        netloc = parsed2.netloc if parsed2 else ""

    # Strip port if present
    if ":" in netloc:
        netloc = netloc.split(":", 1)[0]
    return netloc.lower()


def count_subdomains(domain: str) -> int:
    if not domain:
        return 0
    parts = [p for p in domain.split(".") if p]
    if len(parts) <= 2:
        return 0
    return len(parts) - 2


def has_ip_address(url_clean: str) -> int:
    """Check if URL contains IPv4 or IPv6 literal."""
    text = url_clean
    if IPV4_REGEX.search(text) or IPV6_REGEX.search(text):
        return 1
    return 0


def numeric_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    # Rakam yoğunluğunu sınırlamak için rakam sayısını üstten kırp
    digit_count = sum(ch.isdigit() for ch in text)
    digit_count_clipped = min(digit_count, 5)
    return digit_count_clipped / max(len(text), 1)


def digit_letter_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(ch.isdigit() for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    if letters == 0:
        return 0.0
    return digits / letters


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    # Frequency of each character
    length = len(text)
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def extract_tld(domain: str) -> str:
    if not domain:
        return ""
    parts = [p for p in domain.split(".") if p]
    if not parts:
        return ""
    return parts[-1].lower()


def compute_path_depth(url_clean: str, domain: str) -> int:
    """
    Approximate path depth as the number of '/' segments after the domain.
    """
    if not url_clean:
        return 0
    # Ensure we have something like http://domain/... for urlparse, but guard IPv6 errors
    candidate = url_clean
    if not candidate.lower().startswith(("http://", "https://")):
        candidate = "http://" + candidate
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return 0
    path = parsed.path or ""
    # Remove leading '/'
    while path.startswith("/"):
        path = path[1:]
    if not path:
        return 0
    # Count segments
    segments = [p for p in path.split("/") if p]
    return len(segments)


def is_prestige_tld(tld: str) -> bool:
    return tld in {"com", "org", "net"}


def short_prestige_domain_flag(domain: str, tld: str) -> int:
    """
    Flag domains that are short (3-6 chars) and on common TLDs like .com/.org/.net.
    These are often established brands and can act as a mild safety signal.
    """
    if not domain or not tld:
        return 0
    parts = [p for p in domain.split(".") if p]
    if not parts:
        return 0
    if len(parts) >= 2:
        reg_label = parts[-2]
    else:
        reg_label = parts[0]
    if 3 <= len(reg_label) <= 6 and is_prestige_tld(tld):
        return 1
    return 0


def has_repeated_token(url_clean: str) -> int:
    """
    Detect repeated segments like 'login-login' or 'secure-secure' by
    splitting on common delimiters and checking duplicate tokens.
    """
    if not url_clean:
        return 0
    delimiters = r"[./_\-]"
    tokens = re.split(delimiters, url_clean.lower())
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0
    seen = {}
    for t in tokens:
        seen[t] = seen.get(t, 0) + 1
        if seen[t] >= 2 and len(t) >= 3:
            return 1
    return 0


def brand_spoof_flag(domain: str, url_clean: str) -> int:
    """
    Flag if a popular brand is misused:
    - If the registered domain itself is the brand (örn: google.com, accounts.google.com)
      -> bu GÜVEN işaretidir, spoof olarak SAYILMAZ (0 döner).
    - Eğer marka subdomain veya path içinde geçiyor ve ana domain farklıysa
      (örn: google-login.xyz, secure.google.com.evil.com)
      -> spoof olarak SAYILIR (1 döner).
    """
    text = (url_clean or "").lower()
    dom = (domain or "").lower()

    if not dom:
        return 0

    parts = [p for p in dom.split(".") if p]
    if not parts:
        return 0

    # Registered domain label (ör: google.com, foo.google.co.uk -> "google")
    if len(parts) >= 2:
        reg_label = parts[-2]
    else:
        reg_label = parts[0]

    for brand in BRAND_NAMES:
        if brand not in text:
            continue

        # Gerçek marka domain'i ise (google.com, accounts.google.com vb.) -> güven işareti
        if reg_label == brand:
            continue

        # Marka domain içinde ama ana label farklıysa (google-login.xyz, secure.google.evil.com)
        if brand in dom and reg_label != brand:
            return 1

        # Marka sadece path/query kısmında geçiyorsa da şüpheli
        if brand not in dom and brand in text:
            return 1

    return 0


def extract_features(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    # Ensure URL is string
    df[url_col] = df[url_col].astype(str)

    # Normalized URL (protokol ve www bilgisi tamamen temizlenmiş)
    df["url_clean"] = df[url_col].apply(normalize_url)

    # Structural features (using cleaned URL for ALL length & counts)
    # URL uzunluğunu 80 karakterde tavanla ki aşırı uzun SEO linkleri modeli bozmasın
    df["url_length_raw"] = df["url_clean"].str.len().clip(upper=80)
    df["url_length_clean"] = df["url_clean"].str.len().clip(upper=80)

    df["count_dot"] = df["url_clean"].str.count(r"\.")
    # Tire sayısını maksimum 3 ile sınırla; 20 tire de olsa model 3 görsün
    df["count_hyphen"] = df["url_clean"].str.count("-").clip(upper=3)
    df["count_underscore"] = df["url_clean"].str.count("_")
    df["count_question"] = df["url_clean"].str.count(r"\?")
    df["count_equal"] = df["url_clean"].str.count("=")
    df["has_at"] = df["url_clean"].str.contains("@").astype(int)

    df["numeric_ratio"] = df["url_clean"].apply(numeric_char_ratio)

    # Domain-based features
    df["domain"] = [
        parse_domain(raw, clean)
        for raw, clean in zip(df[url_col].tolist(), df["url_clean"].tolist())
    ]

    df["subdomain_count"] = df["domain"].apply(count_subdomains)
    df["is_shortener"] = df["domain"].isin(SHORTENER_DOMAINS).astype(int)

    # IP address presence
    df["has_ip"] = df["url_clean"].apply(has_ip_address)

    # Suspicious keyword features (0/1)
    url_lower = df["url_clean"].str.lower()
    for kw in SUSPICIOUS_KEYWORDS:
        col_name = f"kw_{kw}"
        df[col_name] = url_lower.str.contains(kw).astype(int)

    # Advanced features
    df["shannon_entropy"] = df["url_clean"].apply(shannon_entropy)
    df["digit_letter_ratio"] = df["url_clean"].apply(digit_letter_ratio)

    df["tld"] = df["domain"].apply(extract_tld)
    df["is_dangerous_tld"] = df["tld"].isin(DANGEROUS_TLDS).astype(int)

    # Eğer domain suffix'i güvenilir TR uzantıları içindeyse, dangerous sinyalini sıfırla
    full_suffixes = []
    for dom in df["domain"].tolist():
        if not dom:
            full_suffixes.append("")
            continue
        parts = [p for p in dom.split(".") if p]
        if len(parts) >= 2:
            suffix = ".".join(parts[-2:])
        else:
            suffix = parts[-1]
        full_suffixes.append(suffix.lower())

    df["tr_suffix"] = full_suffixes
    df.loc[df["tr_suffix"].isin(TRUSTED_TR_SUFFIXES), "is_dangerous_tld"] = 0

    df["is_com_org_net"] = df["tld"].isin({"com", "org", "net"}).astype(int)

    df["path_depth"] = [
        min(compute_path_depth(clean, dom), 5)
        for clean, dom in zip(df["url_clean"].tolist(), df["domain"].tolist())
    ]

    df["short_prestige_domain"] = [
        short_prestige_domain_flag(dom, tld)
        for dom, tld in zip(df["domain"].tolist(), df["tld"].tolist())
    ]

    df["has_repeated_token"] = df["url_clean"].apply(has_repeated_token)

    df["has_brand_spoof"] = [
        brand_spoof_flag(dom, clean)
        for dom, clean in zip(df["domain"].tolist(), df["url_clean"].tolist())
    ]

    return df


def main():
    in_path = "cleaned_sample_data.csv"
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)

    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns 'url' and 'label' in cleaned_sample_data.csv")

    df_features = extract_features(df.copy(), url_col="url")

    # Save original feature set
    out_path_v1 = "features_extracted.csv"
    df_features.to_csv(out_path_v1, index=False)

    # Also save enhanced feature set as v2
    out_path_v2 = "features_extracted_v2.csv"
    df_features.to_csv(out_path_v2, index=False)

    # Report number of feature columns (excluding url, label)
    base_cols = {"url", "label"}
    feature_cols = [c for c in df_features.columns if c not in base_cols]
    print(f"Total features created (excluding 'url' and 'label'): {len(feature_cols)}")
    print("Output saved to", out_path_v1)
    print("Enhanced output saved to", out_path_v2)


if __name__ == "__main__":
    main()

