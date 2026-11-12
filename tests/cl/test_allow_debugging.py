import signal
import sys
import time
import types
from unittest.mock import MagicMock

from cl.util.util import allow_debugging


def _install_fake_debugpy():
    fake_debugpy = types.ModuleType("debugpy")
    fake_debugpy.listen = MagicMock()
    fake_debugpy.wait_for_client = MagicMock()
    fake_debugpy.breakpoint = MagicMock()
    sys.modules["debugpy"] = fake_debugpy
    return fake_debugpy


class TestAllowDebugging:
    def test_defaults_to_loopback(self, monkeypatch):
        """debugpy has no auth of its own, so the default listener must not be
        reachable off box."""
        fake_debugpy = _install_fake_debugpy()
        monkeypatch.setattr(time, "sleep", lambda *_: None)

        registered = {}
        monkeypatch.setattr(
            signal, "signal", lambda sig, handler: registered.setdefault(sig, handler)
        )

        allow_debugging()
        handler = registered[signal.SIGUSR1]
        handler(signal.SIGUSR1, None)

        host, port = fake_debugpy.listen.call_args[0][0]
        assert host == "127.0.0.1"

    def test_explicit_host_override_is_honoured(self, monkeypatch):
        fake_debugpy = _install_fake_debugpy()
        monkeypatch.setattr(time, "sleep", lambda *_: None)

        registered = {}
        monkeypatch.setattr(
            signal, "signal", lambda sig, handler: registered.setdefault(sig, handler)
        )

        allow_debugging(host="0.0.0.0")
        handler = registered[signal.SIGUSR1]
        handler(signal.SIGUSR1, None)

        host, port = fake_debugpy.listen.call_args[0][0]
        assert host == "0.0.0.0"
