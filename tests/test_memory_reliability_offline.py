"""离线执行守卫的契约；不发网络请求、不连接生产库。"""

import socket

import pytest


def test_external_dns_is_blocked_before_any_lookup():
    assert "offline" in socket.getaddrinfo.__name__  # RED 不尝试真实 DNS。
    with pytest.raises(RuntimeError, match="offline"):
        socket.getaddrinfo("offline.invalid", 443)


def test_external_connection_is_blocked_before_any_socket_io():
    assert "offline" in socket.socket.connect.__name__  # RED 不尝试真实连接。
    with socket.socket() as client:
        with pytest.raises(RuntimeError, match="offline"):
            client.connect(("192.0.2.1", 443))


def test_collection_default_database_is_safe():
    from db.base.session_manager import SessionManager
    manager = SessionManager()
    try:
        assert "data/smart_appointment.db" not in str(manager.engine.url).replace("\\", "/")
    finally:
        manager.close()
