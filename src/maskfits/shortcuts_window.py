"""Settings -> "Edit Shortcuts...": lets the user rebind any of the actions
in maskfits.shortcuts.SHORTCUT_ACTIONS to keys, and optionally to an extra
mouse button (Mouse 4 and up) in the right-hand Mouse column. The left/
right/middle button and wheel interactions (see hotkeys_window.
MOUSE_ENTRIES) stay hardcoded into the canvas's own event handlers.

No live preview (unlike the theme editor / accent picker) - rebinding only
takes effect on Save, via the shortcuts_saved signal MaskFitsApp listens for
to rebuild its QShortcut objects (mouse_shortcuts_saved is emitted just
before it, with the mouse bindings).
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from maskfits.shortcuts import (
    SHORTCUT_ACTIONS,
    SHORTCUT_ACTIONS_BY_ID,
    effective_keys,
    find_conflict,
    find_mouse_conflict,
    mouse_binding_label,
)
from maskfits.widgets import MouseCapture, RoundButton, ShortcutCapture

# Fixed so the Mouse column lines up on every row (the key captures before
# it vary between one and two per action). Fits "Shift+Mouse 5 (Forward)".
MOUSE_CAPTURE_WIDTH = 190
DESC_WIDTH = 560
MOUSE_COLUMN_GAP = 10


class ShortcutsWindow(QDialog):
    shortcuts_saved = Signal(dict)
    mouse_shortcuts_saved = Signal(dict)

    def __init__(self, overrides: dict[str, list[str]], parent: Optional[QWidget] = None, *,
                 mouse_bindings: Optional[dict[str, str]] = None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard & Mouse Shortcuts")
        # Non-modal, same as SettingsWindow/ThemeEditorWindow - see
        # SettingsWindow's __init__ docstring comment for the Cocoa
        # modal-nesting bug this avoids.
        self.setModal(False)

        # A working copy - only committed to Settings on Save.
        self.overrides: dict[str, list[str]] = {k: list(v) for k, v in overrides.items()}
        self.mouse_bindings: dict[str, str] = dict(mouse_bindings or {})
        self._captures: dict[str, list[ShortcutCapture]] = {}
        self._mouse_captures: dict[str, MouseCapture] = {}

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

        title = QLabel("Keyboard & Mouse Shortcuts")
        layout.addWidget(title)
        desc = QLabel("Click a key to rebind it, then press the new combination.\n"
                       "A quick Escape tap binds it; hold Escape a second to cancel.\n"
                       "Mouse: click, then press a side button (Mouse 4 and up, modifiers allowed).\n"
                       "Right-click or Backspace clears it; Escape cancels.")
        desc.setWordWrap(True)
        desc.setProperty("dim", True)
        desc.setFixedWidth(DESC_WIDTH)
        layout.addSpacing(2)
        layout.addWidget(desc)

        header = QHBoxLayout()
        header.addStretch(1)
        for text, width in (("Keys", None), ("Mouse", MOUSE_CAPTURE_WIDTH)):
            lbl = QLabel(text)
            lbl.setProperty("dim", True)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if width is not None:
                header.addSpacing(MOUSE_COLUMN_GAP)
                lbl.setFixedWidth(width)
            header.addWidget(lbl)
        layout.addSpacing(8)
        layout.addLayout(header)

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
            mouse_cap = MouseCapture(self.mouse_bindings.get(action.id, ""))
            mouse_cap.setFixedWidth(MOUSE_CAPTURE_WIDTH)
            mouse_cap.bindingChanged.connect(lambda value, a=action.id: self._on_mouse_changed(a, value))
            self._mouse_captures[action.id] = mouse_cap
            row.addSpacing(MOUSE_COLUMN_GAP)
            row.addWidget(mouse_cap)
            layout.addSpacing(8)
            layout.addLayout(row)

        self._conflict_label = QLabel("")
        self._conflict_label.setWordWrap(True)
        self._conflict_label.setFixedWidth(DESC_WIDTH)
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

    def _on_mouse_changed(self, action_id: str, value: str) -> None:
        if value:
            self.mouse_bindings[action_id] = value
        else:
            self.mouse_bindings.pop(action_id, None)
        conflict = find_mouse_conflict(action_id, value, self.mouse_bindings) if value else None
        self._conflict_label.setText(
            f'"{mouse_binding_label(value)}" is already bound to {conflict} too - both will trigger.'
            if conflict else ""
        )

    def _reset_all(self) -> None:
        self.overrides = {}
        self.mouse_bindings = {}
        for action in SHORTCUT_ACTIONS:
            for i, cap in enumerate(self._captures[action.id]):
                cap.setValue(action.defaults[i])
            self._mouse_captures[action.id].setValue("")
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
        self.mouse_shortcuts_saved.emit(dict(self.mouse_bindings))
        self.shortcuts_saved.emit(cleaned)
        self.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            # QDialog's default Escape-closes behavior would fight the
            # ShortcutCapture widgets' own Escape handling (a quick tap
            # binds "Esc" itself; a ~1s hold cancels listening) - either one
            # closing this whole window as a side effect would be a real
            # problem, not just a surprise. Use Cancel to close.
            return
        super().keyPressEvent(event)
