"""Rebindable keyboard shortcuts - the QShortcut-driven ones (see
gui.MaskFitsApp._build_shortcuts) - plus optional extra-mouse-button
bindings for the same actions (see "mouse" below). The left/right/middle
button and wheel interactions (see hotkeys_window.MOUSE_ENTRIES) stay
hardcoded into the canvas's own event handlers.

Each action has one or two DEFAULT key sequences (undo/redo each have a
Ctrl+ combo plus a bare-letter alias). Settings.shortcuts stores only the
actions a user has actually rebound, as {action_id: [sequence, ...]} - so a
future default change for anything the user hasn't touched still applies,
the same reasoning custom_themes.py's per-field storage already follows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ShortcutAction:
    id: str
    label: str
    defaults: tuple[str, ...]


SHORTCUT_ACTIONS: list[ShortcutAction] = [
    # "Ctrl" is Cmd on macOS (Qt maps it). Settings is Ctrl+, rather than
    # Ctrl+S, which is already Quick Export.
    ShortcutAction("open_file", "Open File", ("Ctrl+O",)),
    ShortcutAction("open_settings", "Open Settings", ("Ctrl+,",)),
    ShortcutAction("show_hotkeys", "Show Keyboard Shortcuts", ("Ctrl+K",)),
    ShortcutAction("undo", "Undo", ("Ctrl+Z", "U")),
    ShortcutAction("redo", "Redo", ("Ctrl+Shift+Z", "Y")),
    ShortcutAction("prev_image", "Previous Image", ("Left",)),
    ShortcutAction("next_image", "Next Image", ("Right",)),
    ShortcutAction("reset_mask", "Reset Mask", ("R",)),
    ShortcutAction("grow_shape", "Grow Shape Size", ("E",)),
    ShortcutAction("shrink_shape", "Shrink Shape Size", ("W",)),
    ShortcutAction("reset_shape", "Reset Shape", ("Backspace",)),
    ShortcutAction("cycle_colormap", "Cycle Colormap", ("C",)),
    ShortcutAction("invert_colormap", "Invert Colormap", ("I",)),
    ShortcutAction("toggle_smooth", "Smooth Image (current sigma)", ("S",)),
    ShortcutAction("toggle_bin", "Bin Image (current factor)", ("B",)),
    ShortcutAction("reset_zoom", "Reset Zoom", ("Space",)),
    ShortcutAction("toggle_center_cross", "Toggle Center Cross", ("K",)),
    ShortcutAction("toggle_tool", "Switch Shape (ellipse / satellite)", ("Tab",)),
    ShortcutAction("quick_export", "Quick Export Mask", ("Ctrl+S",)),
    ShortcutAction("prev_extension", "Previous FITS Extension", ("Down",)),
    ShortcutAction("next_extension", "Next FITS Extension", ("Up",)),
    ShortcutAction("digit_1", "Hotkey 1 (ellipticity- / satellite style 1)", ("1",)),
    ShortcutAction("digit_2", "Hotkey 2 (ellipticity+ / satellite style 2)", ("2",)),
    ShortcutAction("digit_3", "Hotkey 3 (angle- / satellite style 3)", ("3",)),
    ShortcutAction("digit_4", "Hotkey 4 (angle+)", ("4",)),
    # "Esc", not "Escape" - QKeySequence's own canonical string for this key
    # (what ShortcutCapture actually produces when the user re-binds
    # something to Escape), so a fresh capture and this default compare
    # equal as plain strings, not just as equivalent QKeySequence objects.
    ShortcutAction("cancel_line", "Cancel Pending Line Click", ("Esc",)),
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


# ---------------------------------------------------------------- mouse

# Extra mouse buttons ("M4", "M5", ... - Back/Forward and any further side
# buttons) can be bound to any action above, optionally with modifiers, as
# e.g. "Mouse4" or "Shift+Mouse5". Button n is the Qt.MouseButton flag with
# bit n-1 set (Back = 0x8 = Mouse4, Forward = 0x10 = Mouse5, up to
# ExtraButton24 = Mouse27). Left/right/middle (1-3) already paint, erase and
# cancel/redo on the canvas, so they aren't bindable. None are bound by
# default; Settings.mouse_shortcuts holds {action_id: binding}.
MOUSE_BUTTON_MIN = 4
MOUSE_BUTTON_MAX = 27
MOUSE_BUTTON_NAMES = {4: "Back", 5: "Forward"}
# Same order Qt writes key-sequence modifiers in; "Ctrl" is Cmd on macOS,
# as for keyboard shortcuts.
MOUSE_MODIFIERS = ("Meta", "Ctrl", "Alt", "Shift")
_MOUSE_BINDING_RE = re.compile(r"^(?:(?:Meta|Ctrl|Alt|Shift)\+)*Mouse(\d+)$")


def mouse_binding(button_value: int, modifiers: tuple[str, ...] = ()) -> Optional[str]:
    """The binding string for a press of the mouse button whose Qt flag
    value is `button_value`, with the named modifiers held - or None for
    the left/right/middle buttons, which aren't bindable."""
    number = int(button_value).bit_length()
    if not MOUSE_BUTTON_MIN <= number <= MOUSE_BUTTON_MAX or button_value & (button_value - 1):
        return None
    held = [m for m in MOUSE_MODIFIERS if m in modifiers]
    return "+".join([*held, f"Mouse{number}"])


def is_mouse_binding(value: object) -> bool:
    if not isinstance(value, str):
        return False
    match = _MOUSE_BINDING_RE.match(value)
    if match is None or not MOUSE_BUTTON_MIN <= int(match.group(1)) <= MOUSE_BUTTON_MAX:
        return False
    mods = value.split("+")[:-1]
    return len(set(mods)) == len(mods)


def mouse_binding_label(binding: str) -> str:
    """Human-readable form: "Shift+Mouse4" -> "Shift+Mouse 4 (Back)"."""
    if not binding:
        return ""
    *mods, button = binding.split("+")
    number = int(button[len("Mouse"):])
    text = f"Mouse {number}"
    if number in MOUSE_BUTTON_NAMES:
        text += f" ({MOUSE_BUTTON_NAMES[number]})"
    return "+".join([*mods, text])


def sanitize_mouse_bindings(raw: object) -> dict[str, str]:
    """Keeps only {known action id: valid binding} - same reasoning as
    sanitize_overrides."""
    if not isinstance(raw, dict):
        return {}
    return {a: b for a, b in raw.items() if a in SHORTCUT_ACTIONS_BY_ID and is_mouse_binding(b)}


def actions_for_mouse(binding: str, bindings: dict[str, str]) -> list[str]:
    """Every action id bound to `binding`, in SHORTCUT_ACTIONS order."""
    return [a.id for a in SHORTCUT_ACTIONS if bindings.get(a.id) == binding]


def find_mouse_conflict(action_id: str, binding: str, bindings: dict[str, str]) -> Optional[str]:
    """Label of another action already bound to the same mouse button, or None."""
    for other in actions_for_mouse(binding, bindings):
        if other != action_id:
            return SHORTCUT_ACTIONS_BY_ID[other].label
    return None


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
