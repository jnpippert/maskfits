"""Coverage for two related features:

- MaskFitsApp(paths, settings=...) applying every Settings-driven default
  (colormap, cut-levels/scale, bin, smooth) at startup - the same path
  run_gui() feeds CLI flags (-c/--cuts/--scale/-b/-s/...) through, merged
  into one Settings object before construction (see cli.py/gui.run_gui).
- The cuts histogram staying fully interactive while IsoPy is the active
  colormap, instead of being locked - a manual edit should stick across
  navigation/re-syncs while IsoPy stays active, and switching away restores
  what was there before IsoPy (not the manual edit).

Passing an explicit `settings=` skips load_settings() entirely (see
MaskFitsApp.__init__), so these tests never touch the real user's QSettings.
"""

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


def _write_fits(path, size=64, mean=100.0, sigma=5.0):
    rng = np.random.default_rng(0)
    data = rng.normal(mean, sigma, size=(size, size)).astype(np.float32)
    fits.PrimaryHDU(data).writeto(path, overwrite=True)


def _write_multi_ext_fits(path):
    """Both HDUs carry real 2D data (unlike an empty-primary-HDU multi-
    extension file), so list_image_extensions()'s own "only HDUs with data"
    filter doesn't override which one ends up active - isolating what
    -x/--extension itself does from that separate, pre-existing fallback."""
    rng = np.random.default_rng(0)
    primary = fits.PrimaryHDU(rng.normal(100.0, 5.0, size=(32, 32)).astype(np.float32))
    primary.header["OBJECT"] = "EXT0"
    ext1 = fits.ImageHDU(rng.normal(100.0, 5.0, size=(32, 32)).astype(np.float32))
    ext1.header["OBJECT"] = "EXT1"
    fits.HDUList([primary, ext1]).writeto(path, overwrite=True)


def test_extension_param_loads_the_requested_hdu(qapp, tmp_path):
    path = tmp_path / "multi.fits"
    _write_multi_ext_fits(path)

    win = MaskFitsApp([str(path)], settings=Settings(), extension=1)
    win.show()
    qapp.processEvents()

    assert win.entry.ext == 1
    assert win.image.header.get("OBJECT") == "EXT1"


def test_extension_param_defaults_to_zero_when_not_given(qapp, tmp_path):
    path = tmp_path / "multi.fits"
    _write_multi_ext_fits(path)

    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    assert win.entry.ext == 0
    assert win.image.header.get("OBJECT") == "EXT0"


def test_extension_param_has_no_impact_on_a_single_extension_file(qapp, tmp_path):
    path = tmp_path / "single.fits"
    _write_fits(path)

    win = MaskFitsApp([str(path)], settings=Settings(), extension=5)
    win.show()
    qapp.processEvents()

    # Entry.ensure_loaded() falls back to the file's own first valid
    # extension when the requested one doesn't exist there.
    assert win.entry.ext == 0
    assert win.image is not None


def test_settings_driven_startup_options_all_apply(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)

    settings = Settings(
        mode="line", zoom=2.0, scale="log", stretch="pct99.5", colormap="IsoPy",
        bin_enabled=True, bin_factor=4, smooth_enabled=True, smooth_sigma=2.0,
    )
    win = MaskFitsApp([str(path)], settings=settings)
    win.show()
    qapp.processEvents()

    assert win.tool == "line"
    assert win.zoom_mult == 2.0
    assert win.scale_function == "log"
    assert win.stretch == "pct99.5"
    assert win.colormap == "IsoPy"
    assert win.entry.is_binned and win.entry.bin_factor == 4
    assert win.entry.is_smoothed and win.entry.smooth_sigma == 2.0


def test_vmin_vmax_override_applies_after_construction(qapp, tmp_path):
    """Mirrors what run_gui() does: vmin/vmax have no Settings field (a
    per-entry value, not a session default), so they're applied directly to
    the just-loaded entry after construction, overriding whatever --cuts/
    the stretch algorithm already computed."""
    path = tmp_path / "img.fits"
    _write_fits(path)

    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    before_lo, before_hi = win.entry.lowcut, win.entry.highcut
    win.entry.lowcut = 90.0
    win.entry.highcut = 110.0
    win._update_cuts_display()
    win.render()

    assert (win.entry.lowcut, win.entry.highcut) != (before_lo, before_hi)
    assert win.entry.lowcut == 90.0
    assert win.entry.highcut == 110.0


def test_isopy_histogram_stays_interactive(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)

    win = MaskFitsApp([str(path)], settings=Settings(colormap="IsoPy"))
    win.show()
    qapp.processEvents()

    assert win.cuts_histogram._lo_entry.isEnabled()
    assert win.cuts_histogram._hi_entry.isEnabled()


def test_isopy_manual_edit_persists_across_resync(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)

    win = MaskFitsApp([str(path)], settings=Settings(colormap="IsoPy"))
    win.show()
    qapp.processEvents()

    win.cuts_histogram._lo_entry.setText("90")
    win.cuts_histogram._on_lo_entry()
    qapp.processEvents()
    assert win.entry.lowcut == 90.0

    # load_current() re-runs _sync_isopy_cuts() - with IsoPy still active,
    # the manual edit must survive, not get reset back to the formula.
    win.load_current(reset_view=False)
    assert win.entry.lowcut == 90.0


def test_cuts_overrides_isopy_default_cuts(qapp, tmp_path):
    """Regression test: IsoPy seeds an entry's lowcut/highcut from its own
    fixed surface-brightness formula the first time it becomes the active
    colormap (see _sync_isopy_cuts) - when IsoPy is the STARTING colormap,
    that seed used to unconditionally win over --cuts even though --cuts
    was applied first, silently discarding it. run_gui() re-applies --cuts
    after construction to fix this (see its own comment) - this checks the
    actual numeric cut values it produces, not just that Settings.stretch
    holds the right string (which was already true before the fix, and is
    exactly why this bug slipped past the less specific end-to-end test
    below the first time around)."""
    from maskfits.imagedata import percentile_cuts

    path = tmp_path / "img.fits"
    _write_fits(path)

    win = MaskFitsApp([str(path)], settings=Settings(colormap="IsoPy", stretch="pct99.5"))
    win.show()
    qapp.processEvents()
    # Mirrors what run_gui() does for an explicit --cuts flag.
    win.entry.apply_stretch("pct99.5")
    win._update_cuts_display()
    win.render()

    expected_lo, expected_hi = percentile_cuts(win.image.data, 99.5)
    assert win.colormap == "IsoPy"
    assert win.entry.lowcut == pytest.approx(expected_lo)
    assert win.entry.highcut == pytest.approx(expected_hi)

    # Leaving IsoPy must restore this same --cuts value, not IsoPy's own
    # formula or a stale pre-fix backup.
    win.set_colormap("Grayscale")
    qapp.processEvents()
    assert win.entry.lowcut == pytest.approx(expected_lo)
    assert win.entry.highcut == pytest.approx(expected_hi)


def _run_cli_and_capture_window(cli_main, argv, monkeypatch):
    """Drives cli.main(argv) with QApplication.exec patched to return
    instead of blocking, and returns the MaskFitsApp it constructed.

    Finds it by diffing QApplication.topLevelWidgets() before/after, rather
    than just taking the first MaskFitsApp instance found - close() hides a
    window but does not destroy it, so an earlier test's (closed) window
    would otherwise still be sitting in topLevelWidgets() and could be
    picked up by mistake instead of the one this call just created."""
    before = {id(w) for w in QApplication.topLevelWidgets()}
    captured = {}

    def fake_exec(self):
        captured["window"] = next(
            w for w in QApplication.topLevelWidgets()
            if isinstance(w, MaskFitsApp) and id(w) not in before
        )
        return 0

    monkeypatch.setattr(QApplication, "exec", fake_exec)
    exit_code = cli_main(argv)
    return exit_code, captured["window"]


def test_cli_main_cuts_overrides_isopy_default_cuts(qapp, tmp_path, monkeypatch):
    """Same regression as test_cuts_overrides_isopy_default_cuts, but
    through the real `maskfits -c isopy --cuts 99.5` argv path (no --vmin/
    --vmax given, unlike test_cli_main_wires_every_new_flag_end_to_end
    below - those would override the cuts value entirely and mask exactly
    this bug, which is exactly why the combined test didn't catch it)."""
    from maskfits.cli import main as cli_main
    from maskfits.imagedata import percentile_cuts

    path = tmp_path / "cli_isopy_cuts.fits"
    _write_fits(path)

    exit_code, win = _run_cli_and_capture_window(
        cli_main, ["-c", "isopy", "--cuts", "99.5", str(path)], monkeypatch,
    )

    assert exit_code == 0
    expected_lo, expected_hi = percentile_cuts(win.image.data, 99.5)
    assert win.colormap == "IsoPy"
    assert win.entry.lowcut == pytest.approx(expected_lo)
    assert win.entry.highcut == pytest.approx(expected_hi)
    win.close()


def test_cli_main_wires_every_new_flag_end_to_end(qapp, tmp_path, monkeypatch):
    """Drives the real argv -> cli.main() -> run_gui() path (QApplication.exec
    patched out so it returns instead of blocking) - the same route a real
    `maskfits -c isopy --cuts 99.5 ...` invocation takes, exercising the
    CLI's mapping tables and the Settings merge together, not just each in
    isolation."""
    from maskfits.cli import main as cli_main

    path = tmp_path / "cli_img.fits"
    _write_fits(path)

    exit_code, win = _run_cli_and_capture_window(
        cli_main,
        ["-c", "isopy", "--cuts", "99.5", "--scale", "log", "-b", "4", "-s", "2",
         "--vmin", "12.5", "--vmax", "34.5", "-m", "s", "-z", "3", str(path)],
        monkeypatch,
    )

    assert exit_code == 0
    assert win.colormap == "IsoPy"
    assert win.stretch == "pct99.5"
    assert win.scale_function == "log"
    assert win.entry.is_binned and win.entry.bin_factor == 4
    assert win.entry.is_smoothed and win.entry.smooth_sigma == 2.0
    assert win.tool == "line"
    assert win.zoom_mult == 3.0
    assert win.entry.lowcut == 12.5
    assert win.entry.highcut == 34.5
    win.close()


def test_isopy_manual_edit_does_not_survive_leaving_and_returning(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)

    win = MaskFitsApp([str(path)], settings=Settings(colormap="IsoPy"))
    win.show()
    qapp.processEvents()

    original_formula_value = win.entry.lowcut
    win.cuts_histogram._lo_entry.setText("90")
    win.cuts_histogram._on_lo_entry()
    qapp.processEvents()
    assert win.entry.lowcut == 90.0

    # Leaving IsoPy restores what was there before entering it - not the
    # manual edit made while inside IsoPy.
    win.set_colormap("Grayscale")
    qapp.processEvents()
    assert win.entry.lowcut != 90.0

    # Re-entering IsoPy re-seeds fresh from the formula, discarding the old
    # manual edit rather than resurrecting it.
    win.set_colormap("IsoPy")
    qapp.processEvents()
    assert win.entry.lowcut == original_formula_value
