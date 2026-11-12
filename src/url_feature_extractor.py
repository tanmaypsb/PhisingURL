"""Extract the UCI Phishing Websites features derivable from a URL string.

No network, DNS, WHOIS, certificate, HTTP response, or page-content checks are
performed. Feature encodings follow the UCI Phishing Websites feature guide:
negative values indicate the suspicious condition for binary features;
URL_Length is 1 (<54 chars), 0 (54–75), or -1 (>75).
"""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit


URL_FEATURES = (
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "port",
    "HTTPS_token",
)

# Exact hostnames of established shortening services. This list is intentionally
# explicit and offline; an unlisted/new service will not be inferred as a shortener.
SHORTENER_HOSTS = frozenset({
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc",
    "lnkd.in", "youtu.be", "rb.gy", "soo.gd", "s.id", "v.gd",
    "short.io", "trib.al", "amzn.to", "ift.tt", "bl.ink", "x.co",
})

FEATURE_DEFINITIONS = {
    "having_IP_Address": "-1 if hostname is an IPv4/IPv6 literal; otherwise 1",
    "URL_Length": "1 if URL length <54; 0 if 54<=length<=75; -1 if length >75",
    "Shortining_Service": "-1 if hostname is a known shortening service; otherwise 1",
    "having_At_Symbol": "-1 if @ occurs in the URL; otherwise 1",
    "double_slash_redirecting": "-1 if URL path contains //; otherwise 1",
    "Prefix_Suffix": "-1 if hostname contains a hyphen; otherwise 1",
    "port": "-1 if explicit port differs from HTTP/HTTPS default; otherwise 1",
    "HTTPS_token": "-1 if hostname contains the token 'https'; otherwise 1",
}


class InvalidURL(ValueError):
    """The submitted string is not a supported absolute HTTP(S) URL."""


def normalize_url(value: str) -> str:
    """Validate an absolute HTTP(S) URL; trim surrounding whitespace only.

    The URL is not fetched or rewritten. A scheme is required to avoid guessing
    whether a bare string is a hostname, path, or another kind of identifier.
    """
    if not isinstance(value, str):
        raise InvalidURL("URL must be entered as text.")
    url = value.strip()
    if not url:
        raise InvalidURL("URL cannot be empty.")
    if any(character.isspace() or ord(character) < 32 for character in url):
        raise InvalidURL("URL cannot contain unescaped spaces or control characters.")
    if "\\" in url:
        raise InvalidURL("URL cannot contain a backslash; use standard forward slashes.")
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        # Accessing .port validates malformed and out-of-range port values.
        _ = parsed.port
    except ValueError as exc:
        raise InvalidURL(f"Malformed URL: {exc}") from exc
    if scheme not in {"http", "https"}:
        raise InvalidURL("Only absolute http:// and https:// URLs are supported.")
    if not parsed.netloc or not hostname:
        raise InvalidURL("URL must include a hostname, for example https://example.com.")
    if "%" in hostname:
        raise InvalidURL("Percent-encoded hostnames are not supported; use a standard hostname or IP address.")
    # Validate DNS-style labels or IP literals. User info is allowed because the
    # UCI having_At_Symbol feature specifically examines the complete URL.
    host = hostname.rstrip(".")
    if hostname.endswith(".."):
        raise InvalidURL("Hostname cannot end with multiple dots.")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        try:
            ascii_host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise InvalidURL("Hostname is not a valid IDN hostname.") from exc
        if len(ascii_host) > 253:
            raise InvalidURL("Hostname exceeds the maximum DNS length.")
        labels = ascii_host.split(".")
        if any(not label or len(label) > 63 or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label) for label in labels):
            raise InvalidURL("Hostname contains an invalid DNS label.")
    return url


def _is_ip_literal(hostname: str) -> bool:
    candidate = hostname.rstrip(".")
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        # The UCI feature guide also describes dotted hexadecimal IPv4
        # components (for example, 0x58.0xCC.0xCA.0x62).
        parts = candidate.split(".")
        if len(parts) == 4:
            try:
                values = [int(part, 16) if part.lower().startswith("0x") else int(part, 10) for part in parts]
                return all(0 <= number <= 255 for number in values) and all(
                    re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)", part) for part in parts
                )
            except ValueError:
                return False
        return False


def _is_shortener(hostname: str) -> bool:
    host = hostname.rstrip(".").encode("idna").decode("ascii").lower()
    return any(host == domain or host.endswith("." + domain) for domain in SHORTENER_HOSTS)


def extract_url_features(value: str) -> dict[str, int]:
    """Return the eight supported UCI features in the trained model's order."""
    url = normalize_url(value)
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    host_lower = hostname.lower()
    length = len(url)
    if length < 54:
        url_length = 1
    elif length <= 75:
        url_length = 0
    else:
        url_length = -1

    default_port = 443 if parsed.scheme.lower() == "https" else 80
    explicit_port = parsed.port
    nonstandard_port = explicit_port is not None and explicit_port != default_port

    features = {
        "having_IP_Address": -1 if _is_ip_literal(hostname) else 1,
        "URL_Length": url_length,
        "Shortining_Service": -1 if _is_shortener(hostname) else 1,
        "having_At_Symbol": -1 if "@" in url else 1,
        "double_slash_redirecting": -1 if "//" in parsed.path else 1,
        "Prefix_Suffix": -1 if "-" in host_lower else 1,
        "port": -1 if nonstandard_port else 1,
        "HTTPS_token": -1 if "https" in host_lower else 1,
    }
    return {name: features[name] for name in URL_FEATURES}
