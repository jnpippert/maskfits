"""The welcome screen painted on the empty image canvas when no file is
open: the app logo as a faint one-color watermark, the name and author,
and a few basic keyboard shortcuts drawn as key caps (live bindings, so a
rebound key shows up here too). See ImageCanvas.paintEvent.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter

from maskfits.shortcuts import effective_keys
from maskfits.theme import Theme

ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "icon.png")

AUTHOR = "Jan-Niklas Pippert"
# (label, shortcut action id) - the rows under the logo.
WELCOME_ROWS = [
    ("Open File", "open_file"),
    ("Settings", "open_settings"),
    ("Keyboard Shortcuts", "show_hotkeys"),
    ("Quick Export Mask", "quick_export"),
    ("Switch Shape", "toggle_tool"),
]

LOGO_MAX = 300
LOGO_MIN = 64  # below this the logo is dropped rather than shown as a speck
WATERMARK_ALPHA = 0.16  # strength of the logo against the canvas
ROW_W = 340
ROW_H = 30
CAP_H = 22
CAP_MIN_W = 24
CAP_GAP = 4

# The icon's dark tile color; anything clearly brighter (the white spiral,
# the red line) is a feature - the only part the watermark keeps.
ICON_TILE_LEVEL = 24
FEATURE_RAMP = 60

MAC_MODIFIERS = {"Meta": "⌃", "Alt": "⌥", "Shift": "⇧", "Ctrl": "⌘"}
MAC_ORDER = ["Meta", "Alt", "Shift", "Ctrl"]  # the order macOS menus use: ⌃⌥⇧⌘

_watermark_cache: dict[tuple, QImage] = {}


def key_caps(sequence: str, mac: Optional[bool] = None) -> list[str]:
    """One key cap per key of a shortcut: "Ctrl+Shift+Z" -> ["Ctrl", "Shift",
    "Z"], or ["⇧", "⌘", "Z"] on macOS (where Qt's "Ctrl" is Command)."""
    if mac is None:
        mac = sys.platform == "darwin"
    text = QKeySequence(sequence).toString(QKeySequence.SequenceFormat.PortableText) or sequence
    text = text.split(", ")[0]  # first chord only
    if text == "+":
        parts = ["+"]
    elif text.endswith("++"):
        parts = text[:-2].split("+") + ["+"]
    else:
        parts = text.split("+")
    modifiers, key = parts[:-1], parts[-1]
    if not mac:
        return modifiers + [key]
    ordered = [m for m in MAC_ORDER if m in modifiers]
    return [MAC_MODIFIERS[m] for m in ordered] + [key]


def watermark_image(color: QColor, size: int) -> Optional[QImage]:
    """The app icon as a one-color watermark, like the editor logo in VS
    Code: only the galaxy and the satellite line, in `color` - the rounded
    tile behind them is left out, so no icon outline shows."""
    key = (color.rgba(), size)
    if key in _watermark_cache:
        return _watermark_cache[key]
    source = QImage(ICON_PATH)
    if source.isNull():
        return None
    source = source.convertToFormat(QImage.Format.Format_RGBA8888).scaled(
        size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    w, h = source.width(), source.height()
    pixels = np.frombuffer(source.constBits(), dtype=np.uint8).reshape(h, source.bytesPerLine())[:, :w * 4]
    pixels = pixels.reshape(h, w, 4).astype(np.float32)
    brightest = pixels[..., :3].max(axis=2)
    feature = np.clip((brightest - ICON_TILE_LEVEL) / FEATURE_RAMP, 0.0, 1.0)
    alpha = pixels[..., 3] / 255.0 * feature * color.alphaF()
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = color.red(), color.green(), color.blue()
    out[..., 3] = np.round(alpha * 255).astype(np.uint8)
    image = QImage(out.tobytes(), w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
    _watermark_cache[key] = image
    return image


def paint_welcome(painter: QPainter, rect: QRectF, theme: Theme, shortcuts: dict[str, list[str]]) -> None:
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    base_font = QFont(painter.font())
    title_font = QFont(base_font)
    title_font.setPointSizeF(base_font.pointSizeF() * 1.9)
    title_font.setWeight(QFont.Weight.DemiBold)

    title_h, author_h, gap = 34.0, 20.0, 26.0
    rows_h = ROW_H * len(WELCOME_ROWS)
    text_h = title_h + author_h + gap + rows_h
    logo = min(LOGO_MAX, rect.height() - text_h - 2 * gap - 40, rect.width() * 0.5)
    if logo < LOGO_MIN:
        logo = 0
    total = (logo + gap if logo else 0) + text_h
    if total > rect.height():
        rows_h, text_h = 0.0, title_h + author_h  # too small: name only
        total = text_h
    y = rect.top() + (rect.height() - total) / 2

    if logo:
        tint = QColor(theme.text)
        tint.setAlphaF(WATERMARK_ALPHA)
        image = watermark_image(tint, int(logo * painter.device().devicePixelRatioF()))
        if image is not None:
            painter.drawImage(QRectF(rect.center().x() - logo / 2, y, logo, logo), image)
        y += logo + gap

    painter.setFont(title_font)
    painter.setPen(QColor(theme.text_dim))
    painter.drawText(QRectF(rect.left(), y, rect.width(), title_h), Qt.AlignmentFlag.AlignCenter, "maskfits")
    y += title_h
    painter.setFont(base_font)
    painter.drawText(QRectF(rect.left(), y, rect.width(), author_h), Qt.AlignmentFlag.AlignCenter, f"by {AUTHOR}")
    y += author_h + gap

    if rows_h:
        left = rect.center().x() - ROW_W / 2
        for label, action_id in WELCOME_ROWS:
            keys = effective_keys(action_id, shortcuts)
            _paint_row(painter, theme, QRectF(left, y, ROW_W, ROW_H), label, key_caps(keys[0]) if keys else [])
            y += ROW_H
    painter.restore()


def _paint_row(painter: QPainter, theme: Theme, row: QRectF, label: str, caps: list[str]) -> None:
    painter.setPen(QColor(theme.text_dim))
    painter.drawText(row, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
    metrics = painter.fontMetrics()
    x = row.right()
    top = row.center().y() - CAP_H / 2
    for cap in reversed(caps):
        width = max(CAP_MIN_W, metrics.horizontalAdvance(cap) + 14)
        x -= width
        cap_rect = QRectF(x, top, width, CAP_H)
        painter.setPen(QColor(theme.panel_border))
        painter.setBrush(QColor(theme.button_bg))
        painter.drawRoundedRect(cap_rect, 4, 4)
        painter.setPen(QColor(theme.text_dim))
        painter.drawText(cap_rect, Qt.AlignmentFlag.AlignCenter, cap)
        x -= CAP_GAP
