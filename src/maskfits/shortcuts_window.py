"""Settings -> "Edit Keyboard Shortcuts...": lets the user rebind any of the
QShortcut-driven actions in maskfits.shortcuts.SHORTCUT_ACTIONS. Mouse-button/
wheel interactions (see hotkeys_window.MOUSE_ENTRIES) aren't covered here -
they're hardcoded into the canvas's own event handlers, not reassignable the
same way.

No live preview (unlike the theme editor / accent picker) - rebinding only
takes effect on Save, via the shortcuts_saved signal MaskFitsApp listens for
to rebuild its QShortcut objects.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from maskfits.shortcuts import SHORTCUT_ACTIONS, SHORTCUT_ACTIONS_BY_ID, effective_keys, find_conflict
from maskfits.widgets import RoundButton, ShortcutCapture


class ShortcutsWindow(QDialog):
    shortcuts_saved = Signal(dict)

    def __init__(self, overrides: dict[str, list[str]], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        # Non-modal, same as SettingsWindow/ThemeEditorWindow - see
        # SettingsWindow's __init__ docstring comment for the Cocoa
        # modal-nesting bug this avoids.
        self.setModal(False)

        # A working copy - only committed to Settings on Save.
        self.overrides: dict[str, list[str]] = {k: list(v) for k, v in overrides.items()}
        self._captures: dict[str, list[ShortcutCapture]] = {}

        self._build()
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

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(0)

        title = QLabel("Keyboard Shortcuts")
        layout.addWidget(title)
        desc = QLabel("Click a key to rebind it, then press the new combination.\n"
                       "A quick Escape tap binds it; hold Escape a second to cancel.")
        desc.setWordWrap(True)
        desc.setProperty("dim", True)
        desc.setFixedWidth(420)
        layout.addSpacing(2)
        layout.addWidget(desc)

        for action in SHORTCUT_ACTIONS:
            row = QHBoxLayout()
            lbl = QLabel(f"{action.label}:")
            lbl.setProperty("dim", True)
            row.addWidget(lbl)
            row.addStretch(1)
            keys = effective_keys(action.id, self.overrides)
            captures = []
            for i in range(len(action.defaults)):
                current = keys[i] if i < len(keys) else ""
                cap = ShortcutCapture(current)
                cap.keySequenceChanged.connect(
                    lambda value, a=action.id, idx=i: self._on_key_changed(a, idx, value)
                )
                captures.append(cap)
                row.addWidget(cap)
            self._captures[action.id] = captures
            layout.addSpacing(8)
            layout.addLayout(row)

        self._conflict_label = QLabel("")
        self._conflict_label.setWordWrap(True)
        self._conflict_label.setFixedWidth(420)
        self._conflict_label.setProperty("state", "warning")
        layout.addSpacing(6)
        layout.addWidget(self._conflict_label)

        layout.addSpacing(10)
        btn_row = QHBoxLayout()
        reset_btn = RoundButton("Reset All To Defaults")
        reset_btn.clicked.connect(self._reset_all)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch(1)
        cancel_btn = RoundButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        save_btn = RoundButton("Save", accent=True)
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    # --------------------------------------------------------------- edits

    def _on_key_changed(self, action_id: str, index: int, value: str) -> None:
        keys = list(effective_keys(action_id, self.overrides))
        keys[index] = value
        self.overrides[action_id] = keys

        conflict = find_conflict(action_id, value, self.overrides)
        self._conflict_label.setText(
            f'"{value}" is already bound to {conflict} too - both will trigger.' if conflict else ""
        )

    def _reset_all(self) -> None:
        self.overrides = {}
        for action in SHORTCUT_ACTIONS:
            for i, cap in enumerate(self._captures[action.id]):
                cap.setValue(action.defaults[i])
        self._conflict_label.setText("")

    # ---------------------------------------------------------------- done

    def _save(self) -> None:
        # Drop any action whose overrides now exactly match its defaults -
        # keeps the persisted set to only genuine rebindings, so a future
        # default change for it still applies (see shortcuts.py docstring).
        cleaned: dict[str, list[str]] = {}
        for action_id, keys in self.overrides.items():
            defaults = list(SHORTCUT_ACTIONS_BY_ID[action_id].defaults)
            if keys != defaults:
                cleaned[action_id] = keys
        self.shortcuts_saved.emit(cleaned)
        self.accept()
