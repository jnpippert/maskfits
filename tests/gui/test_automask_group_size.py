"""Coverage for the auto-mask window's group-size range: off by default,
and when on, keeps only flagged regions between the two slider handles."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(qapp, tmp_path):
    data = np.random.default_rng(1).normal(100.0, 1.0, (120, 120)).astype(np.float32)
    data[10, 10] = 10000.0  # 1px source
    data[40:43, 40:43] = 10000.0  # 9px source
    data[80:90, 80:90] = 10000.0  # 100px source
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(data).writeto(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    win.open_auto_mask()
    dialog = win._auto_mask_window
    dialog.expand_px = 0
    dialog._apply()
    return dialog


def test_off_by_default(dialog):
    assert not dialog.group_size_checkbox.isChecked()
    assert dialog.group_size_enabled is False
    assert dialog._preview[10, 10] and dialog._preview[40, 40] and dialog._preview[85, 85]


def test_ticking_applies_both_limits(dialog):
    dialog.group_size_slider.setValues(5, 50)
    dialog.group_size_checkbox.setChecked(True)
    assert not dialog._preview[10, 10]  # 1px  < min
    assert dialog._preview[40:43, 40:43].all()  # 9px within
    assert not dialog._preview[85, 85]  # 100px > max


def test_moving_the_handles_while_ticked_updates_the_preview(dialog):
    dialog.group_size_checkbox.setChecked(True)
    dialog.group_size_slider.setValues(1, 500)
    assert dialog._preview[10, 10] and dialog._preview[85, 85]
    dialog.group_size_slider.setValues(50, 500)
    assert not dialog._preview[10, 10] and dialog._preview[85, 85]


def test_entries_follow_the_slider_and_vice_versa(dialog):
    dialog.group_size_slider.setValues(3, 120)
    assert (dialog.min_group_entry.text(), dialog.max_group_entry.text()) == ("3", "120")
    dialog.min_group_entry.setText("7")
    dialog.max_group_entry.setText("40")
    dialog.max_group_entry.editingFinished.emit()
    assert dialog.group_size_slider.values() == (7.0, 40.0)
    assert (dialog.min_group_size, dialog.max_group_size) == (7, 40)


def test_reversed_entries_are_swapped(dialog):
    dialog.min_group_entry.setText("90")
    dialog.max_group_entry.setText("10")
    dialog.min_group_entry.editingFinished.emit()
    assert dialog.group_size_slider.values() == (10.0, 90.0)
    assert (dialog.min_group_entry.text(), dialog.max_group_entry.text()) == ("10", "90")


def test_unticking_keeps_everything_again(dialog):
    dialog.group_size_slider.setValues(5, 50)
    dialog.group_size_checkbox.setChecked(True)
    dialog.group_size_checkbox.setChecked(False)
    assert dialog._preview[10, 10] and dialog._preview[85, 85]


# ---------------------------------------------------------- shape + expand


def test_expand_defaults_to_two(qapp, tmp_path):
    path = tmp_path / "e.fits"
    fits.PrimaryHDU(np.zeros((20, 20), dtype=np.float32)).writeto(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.open_auto_mask()
    assert win._auto_mask_window.expand_px == 2


def test_shape_boxes_are_off_by_default(dialog):
    assert dialog.shape_filter is None
    assert not any(box.isChecked() for box in dialog.shape_checkboxes.values())


def test_ticking_one_shape_box_unticks_the_other(dialog):
    dialog.shape_checkboxes["lines"].setChecked(True)
    assert dialog.shape_filter == "lines"
    dialog.shape_checkboxes["circles"].setChecked(True)
    assert dialog.shape_filter == "circles"
    assert not dialog.shape_checkboxes["lines"].isChecked()
    dialog.shape_checkboxes["circles"].setChecked(False)
    assert dialog.shape_filter is None


@pytest.fixture
def streak_dialog(qapp, tmp_path):
    data = np.random.default_rng(2).normal(100.0, 1.0, (150, 150)).astype(np.float32)
    data[30:36, 30:36] = 10000.0  # round-ish source
    data[110, 10:140] = 10000.0  # thin streak
    path = tmp_path / "s.fits"
    fits.PrimaryHDU(data).writeto(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.open_auto_mask()
    dialog = win._auto_mask_window
    dialog.expand_px = 0
    dialog._apply()
    return dialog


def test_lines_only_keeps_the_streak(streak_dialog):
    streak_dialog.shape_checkboxes["lines"].setChecked(True)
    assert streak_dialog._preview[110, 70] and not streak_dialog._preview[32, 32]


def test_circles_only_keeps_the_round_source(streak_dialog):
    streak_dialog.shape_checkboxes["circles"].setChecked(True)
    assert not streak_dialog._preview[110, 70] and streak_dialog._preview[32, 32]


# ------------------------------------------------- background + layout


def _texts(dialog, cls):
    return [w.text() for w in dialog.findChildren(cls)]


def test_background_choices_are_median_and_mode(dialog):
    from maskfits.widgets import RoundButton

    texts = _texts(dialog, RoundButton)
    assert "Median" in texts and "Mode" in texts
    assert dialog.bg_method == "median"
    for gone in ("Constant", "Sigma", "SEM"):
        assert gone not in texts


def test_switching_to_mode_uses_the_mode_background(dialog, monkeypatch):
    from maskfits import automask_window

    calls = []
    real = automask_window.background_stats
    monkeypatch.setattr(automask_window, "background_stats",
                        lambda data, kept, method="median": calls.append(method) or real(data, kept, method))
    dialog._on_bg_method_changed("mode")
    assert dialog.bg_method == "mode"
    assert calls[-1] == "mode"


def test_shape_boxes_are_labelled_satellites_and_galaxies_and_stars(dialog):
    from PySide6.QtWidgets import QLabel

    assert dialog.shape_checkboxes["lines"].text() == "Satellites"
    # "&&" is how Qt shows a literal "&" in a checkbox label.
    assert dialog.shape_checkboxes["circles"].text() == "Galaxies && Stars"
    assert "shape:" not in _texts(dialog, QLabel)


def test_no_neural_network_section(dialog):
    from PySide6.QtWidgets import QLabel

    from maskfits.widgets import RoundButton

    labels = " ".join(_texts(dialog, QLabel) + _texts(dialog, RoundButton)).lower()
    assert "neural" not in labels and "model" not in labels
    assert not hasattr(dialog, "nn_model_path")


# ------------------------------------------------- confirm/discard colors


@pytest.fixture
def restore_theme():
    from maskfits.theme import theme_manager

    manager = theme_manager()
    mode, accent, custom = manager._mode, manager._accent, manager._custom
    yield manager
    manager._custom = custom
    manager.set_mode(mode)
    manager.set_accent(accent)


def test_confirm_and_discard_ignore_a_custom_themes_colors(dialog, restore_theme):
    from dataclasses import replace

    from maskfits.theme import DARK

    restore_theme.set_custom(replace(DARK, green="#ff00ff", danger="#00ffff", danger_hover="#00eeee"))
    confirm, discard = dialog._confirm_btn.styleSheet(), dialog._discard_btn.styleSheet()
    assert DARK.green in confirm and "#ff00ff" not in confirm
    assert DARK.danger in discard and DARK.danger_hover in discard and "#00ffff" not in discard


def test_confirm_and_discard_follow_the_built_in_light_mode(dialog, restore_theme):
    from dataclasses import replace

    from maskfits.theme import LIGHT

    restore_theme.set_custom(replace(LIGHT, green="#ff00ff", danger="#00ffff"))
    assert LIGHT.green in dialog._confirm_btn.styleSheet()
    assert LIGHT.danger in dialog._discard_btn.styleSheet()
