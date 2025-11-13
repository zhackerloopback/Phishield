from urllib.parse import urlparse
import re

def _has_https(url):
    return 1 if url.startswith("https://") else 0

def _count_char(url, ch):
    return url.count(ch)

def _hostname_length(url):
    try:
        netloc = urlparse(url).netloc
        return len(netloc)
    except:
        return 0

def _path_length(url):
    try:
        return len(urlparse(url).path)
    except:
        return 0

def extract_features_from_url(url):
    if not isinstance(url, str):
        url = str(url)
    f1 = _has_https(url)
    f2 = _hostname_length(url)
    f3 = _path_length(url)
    f4 = _count_char(url, "@")
    f5 = _count_char(url, "-")
    f6 = sum(ch.isdigit() for ch in url)
    return [f1, f2, f3, f4, f5, f6]