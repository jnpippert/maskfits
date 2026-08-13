"""Dark color palette and fonts shared across the fitsedit GUI."""

APP_BG = "#141415"
PANEL_BG = "#1d1d1f"
PANEL_BORDER = "#2f2f32"
CANVAS_BG = "#0a0a0b"

TEXT = "#eae7e2"
TEXT_DIM = "#96938d"

ACCENT = "#d97757"
ACCENT_HOVER = "#e39178"
ACCENT_ACTIVE = "#c2603f"

DANGER = "#c1554a"
DANGER_HOVER = "#d16e63"

TRACK = "#3a3a3d"
BUTTON_BG = "#28282b"
BUTTON_HOVER = "#333336"

FONT_FAMILY = "Helvetica"
FONT = (FONT_FAMILY, 11)
FONT_SMALL = (FONT_FAMILY, 10)
FONT_LABEL = (FONT_FAMILY, 10)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
