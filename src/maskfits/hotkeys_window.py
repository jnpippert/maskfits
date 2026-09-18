"""Help -> Hotkeys: a read-only popup listing every mouse interaction (fixed,
not reassignable - hardcoded into the canvas's own event handlers) and every
keyboard shortcut (see maskfits.shortcuts, live-resolved against the current
Settings overrides so a rebound key shows correctly here, not a stale
default).
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from maskfits.shortcuts import SHORTCUT_ACTIONS, effective_keys
from maskfits.widgets import RoundButton

# Mouse-button/wheel interactions - hardcoded into ImageCanvas's own event
# handlers, not backed by a rebindable action the way keyboard shortcuts are.
MOUSE_ENTRIES = [
    ("Left-click / drag", "Paint mask with the current tool"),
    ("Right-click / drag", "Erase mask"),
    ("Middle-click", "Cancel a pending line start point, or redo (satellite mode only)"),
    ("Ctrl + left-click drag", "Pan the view"),
    ("Mouse wheel", "Zoom in / out"),
]

LABEL_WIDTH = 170
# Wide enough that the longest entry ("Cancel a pending line start point, or
# redo (satellite mode only)", ~408px at the app's own font) never wraps to
# a second line - a wrapped QLabel nested in a QHBoxLayout inside a
# QVBoxLayout doesn't reliably reserve the extra row height for it here, so
# a wrap silently overlapped the next row instead of just looking cramped.
DESC_WIDTH = 430


class HotkeysWindow(QDialog):
    def __init__(self, shortcut_overrides: dict[str, list[str]], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Hotkeys")
        # Non-modal, same as SettingsWindow/ThemeEditorWindow - see
        # SettingsWindow's __init__ docstring comment for the Cocoa
        # modal-nesting bug this avoids.
        self.setModal(False)
        self._build(shortcut_overrides)
        self._center_on_parent()

    def _center_on_parent(self) -> None:
        parent = self.parentWidget()
        self.adjustSize()
        if parent is None:
            return
        geo = parent.frameGeometry()
        x = max(geo.x() + (geo.width() - self.width()) // 2, 0)
        y = max(geo.y() + (geo.height() - self.height()) // 2, 0)
        self.move(x, y)

    def _build(self, overrides: dict[str, list[str]]) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(0)

        title = QLabel("Hotkeys")
        layout.addWidget(title)

        mouse_header = QLabel("Mouse")
        mouse_header.setProperty("dim", True)
        layout.addSpacing(8)
        layout.addWidget(mouse_header)
        for keys, desc in MOUSE_ENTRIES:
            self._row(layout, keys, desc)

        kb_header = QLabel("Keyboard - edit these in Settings > Edit Keyboard Shortcuts...")
        kb_header.setProperty("dim", True)
        layout.addSpacing(14)
        layout.addWidget(kb_header)
        for action in SHORTCUT_ACTIONS:
            keys = " / ".join(effective_keys(action.id, overrides))
            self._row(layout, keys, action.label)

        layout.addSpacing(14)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = RoundButton("Close", accent=True)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    @staticmethod
    def _row(layout: QVBoxLayout, keys: str, desc: str) -> None:
        row = QHBoxLayout()
        key_lbl = QLabel(keys)
        key_lbl.setFixedWidth(LABEL_WIDTH)
        row.addWidget(key_lbl)
        desc_lbl = QLabel(desc)
        desc_lbl.setWordWrap(True)
        desc_lbl.setFixedWidth(DESC_WIDTH)
        row.addWidget(desc_lbl, 1)
        layout.addSpacing(5)
        layout.addLayout(row)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            # QDialog's default Escape-closes behavior isn't wanted here,
            # consistent with the other popups in this app - use Close.
            return
        super().keyPressEvent(event)
