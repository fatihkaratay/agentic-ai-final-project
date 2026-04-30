"""
SSL bundle helper.

Some networks intercept TLS to inspect outbound traffic, presenting their
own root CA on connections to api.openai.com. The OpenAI Python SDK uses
httpx, which validates against certifi's bundled CAs only — corporate
roots aren't recognised, and connections fail with CERTIFICATE_VERIFY_FAILED.

If the environment exports SSL_CERT_FILE pointing to a corporate root cert,
this module builds a merged bundle (certifi public CAs + the corporate root)
under .venv/ and returns its path. LLMClient then passes it as httpx's
``verify`` so both public APIs and intercepted endpoints validate cleanly.

Returns None when no corporate cert is configured — the SDK then falls back
to its own default trust store, which is correct for plain networks.
"""

import os
from typing import Optional

import certifi


# Stable cache location next to the active venv.
_REPO_ROOT          = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MERGED_BUNDLE_PATH  = os.path.join(_REPO_ROOT, ".venv", "merged-ca-bundle.pem")


def get_merged_ca_bundle() -> Optional[str]:
    """Return a path to a usable CA bundle, or None to use SDK defaults."""
    corp_cert = os.environ.get("SSL_CERT_FILE")
    if not corp_cert or not os.path.isfile(corp_cert):
        return None

    # If the env already points to certifi or a pre-built merged bundle, reuse it.
    if os.path.abspath(corp_cert) == os.path.abspath(certifi.where()):
        return corp_cert

    certifi_path = certifi.where()
    needs_rebuild = (
        not os.path.isfile(MERGED_BUNDLE_PATH)
        or os.path.getmtime(MERGED_BUNDLE_PATH)
            < max(os.path.getmtime(certifi_path), os.path.getmtime(corp_cert))
    )
    if needs_rebuild:
        os.makedirs(os.path.dirname(MERGED_BUNDLE_PATH), exist_ok=True)
        with open(MERGED_BUNDLE_PATH, "wb") as out:
            for src in (certifi_path, corp_cert):
                with open(src, "rb") as fh:
                    out.write(fh.read())
                    out.write(b"\n")

    return MERGED_BUNDLE_PATH
