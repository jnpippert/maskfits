import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit  # noqa: E402

from maskfits.update_dialog import CODE_WIDTH_MIN, show_update_dialog  # noqa: E402
from maskfits.widgets import RoundButton  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _show_without_blocking(qapp, monkeypatch, **kwargs):
    """show_update_dialog() calls QDialog.exec(), which blocks forever
    waiting for real user interaction - intercept it so the test can
    inspect the built dialog instead of hanging."""
    captured = {}

    def fake_exec(self):
        captured["dialog"] = self
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    show_update_dialog(None, "maskfits", **kwargs)
    return captured["dialog"]


def test_no_code_row_when_not_available(qapp, monkeypatch):
    dialog = _show_without_blocking(
        qapp, monkeypatch, message="You're up to date.", available=False,
        clone_url="https://github.com/jnpippert/maskfits.git",
    )
    assert dialog.findChildren(QLineEdit) == []


def test_no_code_row_without_a_clone_url(qapp, monkeypatch):
    dialog = _show_without_blocking(
        qapp, monkeypatch, message="Update available.", available=True, clone_url=None,
    )
    assert dialog.findChildren(QLineEdit) == []


def test_code_row_shows_the_full_clone_command(qapp, monkeypatch):
    dialog = _show_without_blocking(
        qapp, monkeypatch, message="Update available.", available=True,
        clone_url="https://github.com/jnpippert/maskfits.git",
    )
    entries = dialog.findChildren(QLineEdit)
    assert len(entries) == 1
    assert entries[0].text() == "git clone https://github.com/jnpippert/maskfits.git"
    assert entries[0].isReadOnly()


def test_code_field_widens_for_a_long_url(qapp, monkeypatch):
    long_url = "https://github.com/" + "x" * 100 + "/maskfits.git"
    dialog = _show_without_blocking(
        qapp, monkeypatch, message="Update available.", available=True, clone_url=long_url,
    )
    entry = dialog.findChildren(QLineEdit)[0]
    assert entry.minimumWidth() > CODE_WIDTH_MIN


def test_copy_button_puts_command_on_clipboard(qapp, monkeypatch):
    dialog = _show_without_blocking(
        qapp, monkeypatch, message="Update available.", available=True,
        clone_url="https://github.com/jnpippert/maskfits.git",
    )
    copy_btn = next(b for b in dialog.findChildren(RoundButton) if b.text() == "copy")
    copy_btn.click()
    assert QApplication.clipboard().text() == "git clone https://github.com/jnpippert/maskfits.git"
    assert copy_btn.text() == "copied!"
