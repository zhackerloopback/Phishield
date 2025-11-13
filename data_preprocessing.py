from urllib.parse import urlparse

class PhishingDataProcessor:
    def __init__(self):
        # Keep only the features that we actually compute below
        self.features = [
            "url_length",
            "has_https",
            "special_chars",
            "ip_in_url",
            "suspicious_words",
        ]

    def extract_features(self, url: str):
        """Extract simple, fast URL features for the ML model."""
        url = (url or "").strip()
        parsed = urlparse(url)

        feats = {}

        # 1) URL length
        feats["url_length"] = len(url)

        # 2) HTTPS?
        feats["has_https"] = 1 if url.lower().startswith("https://") else 0

        # 3) Count of special characters
        specials = "@#%&*"
        feats["special_chars"] = sum(1 for ch in url if ch in specials)

        # 4) IP-ish host? (very naive: host made of digits/dots only)
        host = parsed.netloc or parsed.path  # handles cases like "example.com" without scheme
        host_core = host.split(":")[0]
        feats["ip_in_url"] = 1 if host_core and all(c.isdigit() or c == "." for c in host_core) else 0

        # 5) Suspicious keywords
        suspicious_words = ["login", "verify", "secure", "account", "update"]
        low = url.lower()
        feats["suspicious_words"] = sum(1 for w in suspicious_words if w in low)

        return feats
