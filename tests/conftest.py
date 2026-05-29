"""Shared pytest setup.

The suite is meant to run hermetically — $0, no network, no keys — so it must not
read a developer's real ``.env`` (which may point at a live provider). This autouse
fixture pins :class:`daedalus.config.Settings` to OS env + defaults only for the
duration of every test, and clears the cached settings singleton around each one so
fixtures that tweak ``DAE_HOME`` (see ``test_tools``/``test_memory``) take effect.
"""

from __future__ import annotations

import pytest

from daedalus import config


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch):
    monkeypatch.setitem(config.Settings.model_config, "env_file", None)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()
