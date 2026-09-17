"""The popup shown for an update_check.py result - both the manual Help ->
Check New Repo Version... action (git) and the silent startup major-version
check (PyPI) render through show_update_dialog(), so the two look identical.

When an update actually is available, the dialog adds a read-only,
monospace one-line command plus a "copy" button (QApplication's clipboard) -
whatever the caller passes as `command` (a `git clone <url>` for the repo
check, `pip install --upgrade maskfits` for the PyPI check) - a one-click way
to get the exact command rather than making the user retype it from the
message text.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from maskfits.widgets import RoundButton

MESSAGE_WIDTH = 480
# Deliberately a generous per-character estimate rather than measuring the
# actual (possibly font-substituted) QFontMetrics width - on at least the
# offscreen/headless Qt platform, the fixed font that ends up painted isn't
# necessarily the one whose metrics get measured beforehand, which under-
# sized this field enough to visibly truncate a normal GitHub URL.
CODE_PX_PER_CHAR = 9
CODE_WIDTH_MIN = 300
CODE_WIDTH_MAX = 640


def show_update_dialog(parent: Optional[QWidget], title: str, message: str, *,
                        available: bool, command: Optional[str]) -> None:
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setModal(True)

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(16, 14, 16, 16)
    layout.setSpacing(0)

    label = QLabel(message)
    label.setWordWrap(True)
    label.setFixedWidth(MESSAGE_WIDTH)
    layout.addWidget(label)

    # Only worth showing for an actual update - "up to date"/error messages
    # have nothing to copy-paste a command for.
    if available and command:
        layout.addSpacing(12)
        row = QHBoxLayout()

        code_entry = QLineEdit(command)
        code_entry.setReadOnly(True)
        code_entry.setCursorPosition(0)
        code_entry.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        # Wide enough to show the whole command unscrolled for a typical
        # GitHub URL, capped so a pathological one doesn't blow out the
        # dialog - the full command is on the clipboard via "copy" either
        # way, this is just about not truncating the common case.
        fits_width = len(command) * CODE_PX_PER_CHAR + 24
        code_entry.setMinimumWidth(max(min(fits_width, CODE_WIDTH_MAX), CODE_WIDTH_MIN))
        row.addWidget(code_entry, 1)

        copy_btn = RoundButton("copy")

        def do_copy() -> None:
            QApplication.clipboard().setText(command)
            copy_btn.setText("copied!")
            QTimer.singleShot(1500, lambda: copy_btn.setText("copy"))

        copy_btn.clicked.connect(do_copy)
        row.addWidget(copy_btn)
        layout.addLayout(row)

    layout.addSpacing(14)
    btn_row = QHBoxLayout()
    btn_row.addStretch(1)
    ok_btn = RoundButton("ok", accent=True)
    ok_btn.clicked.connect(dialog.accept)
    btn_row.addWidget(ok_btn)
    layout.addLayout(btn_row)

    dialog.exec()
