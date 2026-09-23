"""Regression coverage for a real bug: the toolbar's theme icon didn't
update while cycling themes in an open Settings window, or while editing a
theme's icon in the theme editor - only the COLORS live-previewed (via
theme_manager()'s own signal), because the icon lookup in
gui.MaskFitsApp._refresh_theme_toggle reads self._theme_key, which only
gets updated on an actual commit (Save), not by SettingsWindow's live
preview. Fixed via a parallel icon-preview signal chain
(SettingsWindow.theme_icon_previewed / ThemeEditorWindow.icon_previewed)
that both dialogs emit live and gui.MaskFitsApp._preview_theme_icon applies
directly to the toolbar button, bypassing the stale self._theme_key lookup
entirely until something is actually saved."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402
from maskfits.theme_editor_window import ThemeEditorWindow  # noqa: E402

COLORS = {
    "mode": "dark", "app_bg": "#101010", "panel_bg": "#202020", "panel_border": "#303030",
    "text": "#eeeeee", "text_dim": "#999999", "accent": "#00aa66", "danger": "#cc3333",
    "green": "#33cc66", "warning": "#ccaa33", "blue": "#3366cc", "track": "#404040",
    "button_bg": "#282828", "canvas_bg": "#000000",
}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def themes(monkeypatch):
    """A fake custom-theme store, patched into every module that imports
    these names directly (custom_themes itself - for theme_icon()'s own
    internal lookups - plus settings_window and theme_editor_window, which
    each import separate bound copies of the same functions)."""
    store: dict[str, dict] = {}
    fake_list = lambda s=None: store  # noqa: E731
    fake_get = lambda name, s=None: store.get(name)  # noqa: E731
    fake_save = lambda name, colors, s=None: store.update({name: colors})  # noqa: E731

    for target in ("maskfits.custom_themes", "maskfits.settings_window", "maskfits.theme_editor_window"):
        monkeypatch.setattr(f"{target}.list_custom_themes", fake_list, raising=False)
        monkeypatch.setattr(f"{target}.get_custom_theme", fake_get, raising=False)
        monkeypatch.setattr(f"{target}.save_custom_theme", fake_save, raising=False)
    return store


def _forest(icon="\U0001F332"):
    return {**COLORS, "icon": icon}


def test_picking_a_custom_theme_in_the_combo_updates_the_toolbar_live(qapp, themes):
    themes["Forest"] = _forest()
    win = MaskFitsApp([], settings=Settings(theme="dark"))
    win.show()
    assert win.theme_toggle.text() != "\U0001F332"

    win._open_settings()
    dlg = win._settings_window
    dlg._reload_theme_combo(select="dark")
    dlg._apply_theme_choice("Forest")

    assert win.theme_toggle.text() == "\U0001F332"


def test_picking_dark_or_light_in_the_combo_updates_the_toolbar_live(qapp, themes):
    themes["Forest"] = _forest()
    win = MaskFitsApp([], settings=Settings(theme="Forest"))
    win.show()
    assert win.theme_toggle.text() == "\U0001F332"

    win._open_settings()
    dlg = win._settings_window
    dlg._apply_theme_choice("light")

    assert win.theme_toggle.text() == "☀️"


def test_editing_a_themes_icon_updates_the_toolbar_live_before_save(qapp, themes):
    themes["Forest"] = _forest()
    win = MaskFitsApp([], settings=Settings(theme="Forest"))
    win.show()
    assert win.theme_toggle.text() == "\U0001F332"

    win._open_settings()
    dlg = win._settings_window
    dlg._edit_theme()
    editor = dlg._theme_editor
    editor._icon_picker._on_picked("\U0001F30A")

    assert win.theme_toggle.text() == "\U0001F30A"
    # Not persisted yet - the store still has the old icon.
    assert themes["Forest"]["icon"] == "\U0001F332"


def test_canceling_the_editor_reverts_the_toolbar_to_the_saved_icon(qapp, themes):
    themes["Forest"] = _forest()
    win = MaskFitsApp([], settings=Settings(theme="Forest"))
    win.show()

    win._open_settings()
    dlg = win._settings_window
    dlg._edit_theme()
    editor = dlg._theme_editor
    editor._icon_picker._on_picked("\U0001F30A")
    assert win.theme_toggle.text() == "\U0001F30A"

    editor._cancel()

    assert win.theme_toggle.text() == "\U0001F332"


def test_canceling_settings_reverts_the_toolbar_to_the_original_icon(qapp, themes):
    themes["Forest"] = _forest()
    win = MaskFitsApp([], settings=Settings(theme="dark"))
    win.show()
    original = win.theme_toggle.text()

    win._open_settings()
    dlg = win._settings_window
    dlg._reload_theme_combo(select="dark")
    dlg._apply_theme_choice("Forest")
    assert win.theme_toggle.text() == "\U0001F332"

    dlg._cancel()

    assert win.theme_toggle.text() == original


def test_saving_a_new_icon_keeps_the_toolbar_showing_it(qapp, themes):
    themes["Forest"] = _forest()
    win = MaskFitsApp([], settings=Settings(theme="Forest"))
    win.show()

    win._open_settings()
    dlg = win._settings_window
    dlg._edit_theme()
    editor = dlg._theme_editor
    editor._icon_picker._on_picked("\U0001F30A")
    editor._save()

    assert win.theme_toggle.text() == "\U0001F30A"
    assert themes["Forest"]["icon"] == "\U0001F30A"


def test_new_theme_editor_icon_pick_also_previews_live(qapp, themes):
    """Even for a brand-new (not-yet-saved, not-yet-selected) theme, the
    editor already live-previews its COLORS via theme_manager() - the icon
    should track the same "New Theme..." draft-in-progress live too."""
    win = MaskFitsApp([], settings=Settings(theme="dark"))
    win.show()

    win._open_settings()
    dlg = win._settings_window
    dlg._new_theme()
    editor = dlg._theme_editor
    editor._icon_picker._on_picked("\U0001F984")  # unicorn

    assert win.theme_toggle.text() == "\U0001F984"
