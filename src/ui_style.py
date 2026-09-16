"""Shared look-and-feel for the Qt apps in this repo.

Extracted verbatim from low_light_app.py (2026-09, Phase 0 of the lightbox
calibration build — see docs/lightbox-calibration-plan.md) so
lightbox_calibration.py can reuse the same stylesheet and status colours
instead of forking them. low_light_app.py re-imports everything here; its
behaviour is unchanged.
"""
from __future__ import annotations

import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Stylesheet — adapted from 28_BK-8542B_IVCurve/BK-8542B_IVCurve/src/IV_app.py
# ---------------------------------------------------------------------------

_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #f0f2f5;
    color: #1a1a1a;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 11pt;
}
QPushButton {
    background-color: #ffffff;
    color: #1a1a1a;
    border: 2px solid #9aa0a6;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 12pt;
    font-weight: bold;
    min-height: 34px;
}
QPushButton:hover { background-color: #e8eaed; border-color: #444; }
QPushButton:pressed { background-color: #d2d4d8; }
QPushButton:disabled { color: #aaaaaa; border-color: #d0d0d0; background-color: #f0f0f0; }

QPushButton[primary="true"] {
    background-color: #1a56db;
    border-color: #1246c0;
    color: #ffffff;
}
QPushButton[primary="true"]:hover { background-color: #1246c0; border-color: #0d3d9e; }
QPushButton[primary="true"]:disabled { background-color: #c5d5f5; border-color: #a8bfec; color: #6685c5; }

QPushButton[danger="true"] {
    background-color: #c53030;
    border-color: #9b1c1c;
    color: #ffffff;
}
QPushButton[danger="true"]:hover { background-color: #9b1c1c; border-color: #7b1212; }

QLabel { background: transparent; }
QLineEdit {
    background-color: #ffffff;
    border: 2px solid #9aa0a6;
    border-radius: 4px;
    padding: 4px 6px;
    font-size: 11pt;
}
QComboBox {
    background-color: #ffffff;
    border: 2px solid #9aa0a6;
    border-radius: 4px;
    padding: 4px 6px;
    font-size: 11pt;
}

QTableWidget {
    background-color: #ffffff;
    color: #1a1a1a;
    gridline-color: #c0c4cc;
    selection-background-color: #1a56db;
    selection-color: #ffffff;
    font-size: 10pt;
    border: 1px solid #9aa0a6;
}
QHeaderView::section {
    background-color: #e8eaed;
    color: #1a1a1a;
    font-weight: bold;
    font-size: 10pt;
    padding: 6px 4px;
    border: 1px solid #9aa0a6;
}
QTableWidget::item { padding: 4px; }

QProgressBar {
    background-color: #e8eaed;
    border: 2px solid #9aa0a6;
    border-radius: 5px;
    min-height: 22px;
    text-align: center;
    color: #1a1a1a;
    font-weight: bold;
    font-size: 10pt;
}
QProgressBar::chunk { background-color: #1a56db; border-radius: 3px; }

QSplitter::handle { background: #c0c4cc; }
"""

_STAGE_STYLE = {
    "info": "background:#e8eaed; color:#333333; border:1px solid #9aa0a6; border-radius:5px; padding:8px; font-weight:bold;",
    "success": "background:#e6f4ea; color:#1b5e20; border:1px solid #34a853; border-radius:5px; padding:8px; font-weight:bold;",
    "warning": "background:#fff4e5; color:#8a5300; border:1px solid #f2a900; border-radius:5px; padding:8px; font-weight:bold;",
    "error": "background:#fdeaea; color:#9b1c1c; border:1px solid #c53030; border-radius:5px; padding:8px; font-weight:bold;",
}


def open_in_explorer(path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
