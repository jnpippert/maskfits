"""Shared test setup.

Every test gets its own empty settings location, so nothing a test saves
(Settings window Save, theme editor, shortcut editor, ...) can reach the
real user config in ~/.config/maskfits - it used to, and running the suite
reset the user's own defaults.
"""

import pytest
from PySide6.QtCore import QSettings


@pytest.fixture(autouse=True)
def isolated_settings_store(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("qsettings"))
    for fmt in (QSettings.Format.IniFormat, QSettings.Format.NativeFormat):
        for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
            QSettings.setPath(fmt, scope, path)
    yield path
