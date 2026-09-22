"""Rebindable keyboard shortcuts - the QShortcut-driven ones (see
gui.MaskFitsApp._build_shortcuts), as opposed to mouse-button/wheel
interactions (see hotkeys_window.MOUSE_ENTRIES), which are hardcoded into
the canvas's own event handlers and aren't reassignable the same way.

Each action has one or two DEFAULT key sequences (undo/redo each have a
Ctrl+ combo plus a bare-letter alias). Settings.shortcuts stores only the
actions a user has actually rebound, as {action_id: [sequence, ...]} - so a
future default change for anything the user hasn't touched still applies,
the same reasoning custom_themes.py's per-field storage already follows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ShortcutAction:
    id: str
    label: str
    defaults: tuple[str, ...]


SHORTCUT_ACTIONS: list[ShortcutAction] = [
    ShortcutAction("undo", "Undo", ("Ctrl+Z", "U")),
    ShortcutAction("redo", "Redo", ("Ctrl+Shift+Z", "Y")),
    ShortcutAction("prev_image", "Previous image", ("Left",)),
    ShortcutAction("next_image", "Next image", ("Right",)),
    ShortcutAction("reset_mask", "Reset mask", ("R",)),
    ShortcutAction("grow_shape", "Grow shape size", ("E",)),
    ShortcutAction("shrink_shape", "Shrink shape size", ("W",)),
    ShortcutAction("cycle_colormap", "Cycle colormap", ("C",)),
    ShortcutAction("invert_colormap", "Invert colormap", ("I",)),
    ShortcutAction("toggle_smooth", "Smooth image (current sigma)", ("S",)),
    ShortcutAction("toggle_bin", "Bin image (current factor)", ("B",)),
    ShortcutAction("reset_zoom", "Reset zoom", ("Ctrl+R",)),
    ShortcutAction("prev_extension", "Previous FITS extension", ("Down",)),
    ShortcutAction("next_extension", "Next FITS extension", ("Up",)),
    ShortcutAction("digit_1", "Hotkey 1 (ellipticity- / satellite style 1)", ("1",)),
    ShortcutAction("digit_2", "Hotkey 2 (ellipticity+ / satellite style 2)", ("2",)),
    ShortcutAction("digit_3", "Hotkey 3 (angle- / satellite style 3)", ("3",)),
    ShortcutAction("digit_4", "Hotkey 4 (angle+)", ("4",)),
    # "Esc", not "Escape" - QKeySequence's own canonical string for this key
    # (what ShortcutCapture actually produces when the user re-binds
    # something to Escape), so a fresh capture and this default compare
    # equal as plain strings, not just as equivalent QKeySequence objects.
    ShortcutAction("cancel_line", "Cancel a pending line click", ("Esc",)),
]

SHORTCUT_ACTIONS_BY_ID: dict[str, ShortcutAction] = {a.id: a for a in SHORTCUT_ACTIONS}


def effective_keys(action_id: str, overrides: dict[str, list[str]]) -> tuple[str, ...]:
    """The actual key sequence(s) bound to `action_id` right now - the
    user's own rebinding if they've set one, else the action's default."""
    override = overrides.get(action_id)
    if override is not None:
        return tuple(override)
    return SHORTCUT_ACTIONS_BY_ID[action_id].defaults


def sanitize_overrides(raw: dict) -> dict[str, list[str]]:
    """Drops anything not shaped like {known_action_id: [str, ...]} - a
    stale action id (removed since), a non-list value, or a non-string
    entry - rather than letting garbage from a hand-edited or old-version
    settings file propagate into _build_shortcuts()."""
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, list[str]] = {}
    for action_id, keys in raw.items():
        if action_id not in SHORTCUT_ACTIONS_BY_ID:
            continue
        if not isinstance(keys, list) or not all(isinstance(k, str) and k for k in keys):
            continue
        if keys:
            cleaned[action_id] = keys
    return cleaned


def find_conflict(action_id: str, key: str, overrides: dict[str, list[str]]) -> Optional[str]:
    """If `key` is already bound to some OTHER action (given the current
    overrides), returns that action's label - else None. Used by the
    shortcut editor to warn about (not block) a duplicate binding."""
    for other in SHORTCUT_ACTIONS:
        if other.id == action_id:
            continue
        if key in effective_keys(other.id, overrides):
            return other.label
    return None
