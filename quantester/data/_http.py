"""Shared HTTP + API-key helpers for REST market-data adapters.

Requires the optional dependency:  pip install "quantester[data]"
(which pulls in ``requests``). Adapters raise an actionable ImportError when
``requests`` is missing.
"""

from __future__ import annotations

import os
from typing import Any


def import_requests():
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "This data/macro feed requires requests: "
            "pip install 'quantester[data]'"
        ) from exc
    return requests


def resolve_api_key(
    api_key: str | None,
    *,
    env_var: str,
    required: bool = True,
    provider: str = "provider",
) -> str | None:
    """Prefer an explicit ``api_key``; else read ``env_var`` from the environment.

    When ``required`` is True and neither source yields a non-empty key, raise
    ValueError with the env-var name so researchers know what to set.
    """
    if api_key is not None and str(api_key).strip():
        return str(api_key).strip()
    env = os.environ.get(env_var, "").strip()
    if env:
        return env
    if required:
        raise ValueError(
            f"{provider} requires an API key: pass api_key=... or set "
            f"the {env_var} environment variable."
        )
    return None


def http_get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
):
    """GET ``url`` and return the ``requests.Response`` (raises for HTTP errors)."""
    requests = import_requests()
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def http_get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> Any:
    return http_get(url, params=params, headers=headers, timeout=timeout).json()


def http_get_text(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> str:
    return http_get(url, params=params, headers=headers, timeout=timeout).text
