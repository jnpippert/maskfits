import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402

from maskfits.settings import (  # noqa: E402
    Settings,
    format_mask_filename,
    is_valid_export_filename_format,
    load_settings,
    save_settings,
)


@pytest.fixture
def store(tmp_path):
    # An isolated INI-backed QSettings per test, never the real user config.
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def test_load_settings_defaults_when_nothing_saved(store):
    settings = load_settings(store)
    assert settings == Settings()


def test_save_then_load_round_trips(store):
    original = Settings(
        theme="light", mode="line", colormap="Viridis", scale="log", stretch="pct99.5",
        bin_enabled=True, bin_factor=5, smooth_enabled=True, smooth_sigma=2.5,
        export_dir="cwd", zoom=3.0, accent_color="#00aabb",
    )
    save_settings(original, store)
    loaded = load_settings(store)
    assert loaded == original


def test_load_settings_falls_back_on_invalid_stored_values(store):
    store.setValue("theme", "not-a-real-theme")
    store.setValue("mode", "bogus")
    store.setValue("colormap", "NoSuchColormap")
    store.setValue("scale", "bogus")
    store.setValue("stretch", "bogus")
    store.setValue("export_dir", "bogus")
    store.sync()

    settings = load_settings(store)
    defaults = Settings()
    assert settings.theme == defaults.theme
    assert settings.mode == defaults.mode
    assert settings.colormap == defaults.colormap
    assert settings.scale == defaults.scale
    assert settings.stretch == defaults.stretch
    assert settings.export_dir == defaults.export_dir


@pytest.mark.parametrize("stretch", ["minmax", "zscale", "pct99.5", "pct99.0", "pct98.0", "pct95.0", "pct90.0"])
def test_all_stretch_choices_round_trip(store, stretch):
    save_settings(Settings(stretch=stretch), store)
    assert load_settings(store).stretch == stretch


def test_accent_color_none_round_trips_as_none(store):
    save_settings(Settings(accent_color=None), store)
    assert load_settings(store).accent_color is None


def test_theme_accepts_a_registered_custom_theme_name(store):
    from maskfits.custom_themes import save_custom_theme

    save_custom_theme("MyTheme", {"mode": "dark", "app_bg": "#000000"}, store)
    save_settings(Settings(theme="MyTheme"), store)
    assert load_settings(store).theme == "MyTheme"


def test_shortcuts_round_trip(store):
    overrides = {"undo": ["Ctrl+Z"], "reset_mask": ["X"]}
    save_settings(Settings(shortcuts=overrides), store)
    assert load_settings(store).shortcuts == overrides


def test_shortcuts_default_to_empty(store):
    assert load_settings(store).shortcuts == {}


def test_shortcuts_json_sanitizes_garbage_on_load(store):
    store.setValue("shortcuts_json", '{"undo": ["Z"], "bogus_action": ["Q"], "redo": "not-a-list"}')
    store.sync()
    assert load_settings(store).shortcuts == {"undo": ["Z"]}


def test_shortcuts_json_malformed_json_falls_back_to_empty(store):
    store.setValue("shortcuts_json", "{not valid json")
    store.sync()
    assert load_settings(store).shortcuts == {}


# ----------------------------------------------------- export filename format


def test_export_filename_format_defaults_to_mask_filename(store):
    assert load_settings(store).export_filename_format == "mask_$FILENAME"


def test_export_filename_format_round_trips(store):
    save_settings(Settings(export_filename_format="$FILENAME_out"), store)
    assert load_settings(store).export_filename_format == "$FILENAME_out"


def test_export_filename_format_falls_back_when_missing_the_placeholder(store):
    store.setValue("export_filename_format", "no_placeholder_here")
    store.sync()
    assert load_settings(store).export_filename_format == "mask_$FILENAME"


def test_export_filename_format_falls_back_when_empty(store):
    store.setValue("export_filename_format", "")
    store.sync()
    assert load_settings(store).export_filename_format == "mask_$FILENAME"


def test_is_valid_export_filename_format():
    assert is_valid_export_filename_format("mask_$FILENAME") is True
    assert is_valid_export_filename_format("$FILENAME") is True
    assert is_valid_export_filename_format("") is False
    assert is_valid_export_filename_format("no_placeholder") is False


def test_format_mask_filename_uses_the_default_format():
    assert format_mask_filename("mask_$FILENAME", "img42") == "mask_img42.fits"


def test_format_mask_filename_adds_fits_suffix_when_missing():
    assert format_mask_filename("$FILENAME_masked", "img42") == "img42_masked.fits"


def test_format_mask_filename_does_not_double_the_suffix():
    assert format_mask_filename("$FILENAME.fits", "img42") == "img42.fits"


@pytest.mark.parametrize("suffix", [".fits", ".FITS", ".fit", ".fts", ".Fit"])
def test_format_mask_filename_recognizes_every_fits_suffix_case_insensitively(suffix):
    result = format_mask_filename(f"$FILENAME{suffix}", "img42")
    assert result == f"img42{suffix}"


def test_format_mask_filename_replaces_every_occurrence():
    assert format_mask_filename("$FILENAME_a_$FILENAME_b", "x") == "x_a_x_b.fits"
