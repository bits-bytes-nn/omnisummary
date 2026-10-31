import socket

import httpx
import pytest


def test_no_test_can_resolve_or_connect():
    # conftest's last line of defence: a missed stub must fail here, not reach a real service.
    with pytest.raises(RuntimeError, match="Network access is disabled"):
        socket.getaddrinfo("graph.threads.net", 443)
    with pytest.raises(RuntimeError, match="Network access is disabled"):
        httpx.get("https://graph.threads.net/v1.0/me", timeout=1)
