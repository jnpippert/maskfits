"""Tests must never read or write the real user config (see conftest.py)."""

import os

from maskfits import custom_themes, settings


def test_settings_store_is_not_the_users_config(isolated_settings_store):
    real_config = os.path.expanduser("~/.config")
    for store in (settings._backing_store(), custom_themes._store()):
        path = os.path.realpath(store.fileName())
        assert path.startswith(os.path.realpath(isolated_settings_store))
        assert not path.startswith(os.path.realpath(real_config))
