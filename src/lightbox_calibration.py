#!/usr/bin/env python3
"""Lightbox spatial calibration tool -- standalone PyQt6 GUI.

Collects lux/irradiance at every node of a scored grid, at each operator-set
variac level, to compute a per-level correction factor K (panel-footprint
mean / sensor-node value). See docs/lightbox-calibration-plan.md for the
full spec. All measurement/analysis maths lives in calibration_model.py
(zero Qt); this file is the GUI + capture/flow-control layer on top of it.

Run with --simulate to exercise the whole app (GUI, CSV writer, resume path,
analysis) with a synthetic light field and no hardware attached.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import serial.tools.list_ports

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibration_model as cm
import sensors
from ui_style import _STYLESHEET, _STAGE_STYLE, open_in_explorer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "calibration"

AMBER_3SIGMA_PCT = 5.0  # GUI-level "noisy capture" warning threshold (plan 7.3)
CAPTURE_SAMPLE_INTERVAL_S = 0.05


def _log(level: str, event: str, **fields) -> None:
    sensors._log(level, event, **fields)


# ---------------------------------------------------------------------------
# Simulation mode: synthetic light field (plan section 8.1).
#
# A plausible cosine-falloff profile centred on the plate, scaled by the
# entered VAC, with ~1% Gaussian noise and a slow 2% linear drift over a
# pass. Distinct base magnitudes for lux vs irradiance so the lux/irr ratio
# map (section 6.5) has something non-trivial to show. This has no bearing
# on the real hardware code path at all -- it only feeds the same LiveValue
# / TimeSeriesBuffer / HistoryIrradiance objects a real sensor thread would.
# ---------------------------------------------------------------------------

_SIM_BASE = {"lux": 220.0, "irr": 170.0}


def simulate_field_value(x_mm: float, y_mm: float, grid: cm.GridConfig, vac: float, quantity: str, drift_fraction: float) -> float:
    x_centre = (grid.cols - 1) * grid.pitch_mm / 2.0
    y_centre = (grid.rows - 1) * grid.pitch_mm / 2.0
    dist = math.hypot(x_mm - x_centre, y_mm - y_centre)
    max_dist = math.hypot(x_centre, y_centre) or 1.0
    falloff = math.cos((math.pi / 2.0) * min(dist / max_dist, 1.0))
    value = _SIM_BASE[quantity] * max(vac, 0.0) / 50.0 * falloff * (1.0 + 0.02 * drift_fraction)
    if value > 0:
        value += random.gauss(0.0, 0.01 * value)
    return max(value, 0.0)


# ---------------------------------------------------------------------------
# Grid widget -- 11x7 (configurable) cells drawn to scale, used both by the
# Setup tab (pick reference/sensor/footprint nodes) and the Run tab (capture
# progress). Zero business logic beyond node<->pixel mapping; state is fed
# in by the caller.
# ---------------------------------------------------------------------------

_STATE_COLORS = {
    "pending": QColor("#d6dade"),
    "covered": QColor("#cfe8d8"),
    "captured": QColor("#34a853"),
    "flagged": QColor("#f2a900"),
    "skipped": QColor("#b0b4b8"),
}


class GridWidget(QWidget):
    def __init__(self, grid: cm.GridConfig, parent=None, clickable: bool = True):
        super().__init__(parent)
        self.grid = grid
        self.clickable = clickable
        self.cell_state: dict[str, str] = {}
        self.cell_value: dict[str, float] = {}
        self.hover_text: dict[str, str] = {}
        self.redone_nodes: set[str] = set()
        self.covered_nodes: set[str] = set()
        self.sensor_node: str | None = None
        self.reference_node: str | None = None
        self.current_node: str | None = None
        self.on_node_clicked = None  # callable(node_label), left click
        self.on_node_right_clicked = None  # callable(node_label), right click
        self.setMouseTracking(True)
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_grid(self, grid: cm.GridConfig) -> None:
        self.grid = grid
        self.reset_cells()

    def reset_cells(self) -> None:
        self.cell_state.clear()
        self.cell_value.clear()
        self.hover_text.clear()
        self.redone_nodes.clear()
        self.update()

    def _cell_rects(self):
        w, h = self.width(), self.height()
        margin = 4
        cw = (w - 2 * margin) / self.grid.cols
        ch = (h - 2 * margin) / self.grid.rows
        for row_index in range(self.grid.rows):
            for col_index in range(self.grid.cols):
                x = margin + col_index * cw
                y = margin + row_index * ch
                yield col_index, row_index, x, y, cw, ch

    def _node_at(self, pos) -> str | None:
        for col_index, row_index, x, y, cw, ch in self._cell_rects():
            if x <= pos.x() <= x + cw and y <= pos.y() <= y + ch:
                return cm.node_label(col_index, row_index)
        return None

    def mousePressEvent(self, event) -> None:
        if not self.clickable:
            return
        label = self._node_at(event.position())
        if not label:
            return
        if event.button() == Qt.MouseButton.RightButton:
            if self.on_node_right_clicked:
                self.on_node_right_clicked(label)
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.on_node_clicked:
                self.on_node_clicked(label)

    def mouseMoveEvent(self, event) -> None:
        label = self._node_at(event.position())
        text = self.hover_text.get(label) if label else None
        if text:
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
        else:
            QToolTip.hideText()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        values = [v for v in self.cell_value.values() if v is not None]
        vmin, vmax = (min(values), max(values)) if values else (0.0, 1.0)
        vspan = (vmax - vmin) or 1.0

        for col_index, row_index, x, y, cw, ch in self._cell_rects():
            label = cm.node_label(col_index, row_index)
            state = self.cell_state.get(label, "covered" if label in self.covered_nodes else "pending")
            color = QColor(_STATE_COLORS.get(state, _STATE_COLORS["pending"]))

            if state == "captured":
                value = self.cell_value.get(label)
                if value is not None:
                    frac = 0.35 + 0.65 * (value - vmin) / vspan
                    color.setAlphaF(max(0.15, min(1.0, frac)))

            painter.fillRect(int(x), int(y), int(cw), int(ch), color)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#9aa0a6"), 1))
            painter.drawRect(int(x), int(y), int(cw), int(ch))

            painter.setPen(QPen(QColor("#1a1a1a")))
            painter.drawText(int(x) + 3, int(y) + 12, label)

            if label in self.redone_nodes:
                painter.setBrush(QColor("#1a56db"))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(int(x + cw - 9), int(y + 3), 6, 6)

            if label == self.reference_node:
                painter.setBrush(QColor("#0d3d9e"))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(int(x + 3), int(y + ch - 9), 6, 6)

            if label == self.sensor_node:
                painter.setBrush(QColor("#c53030"))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(int(x + cw - 9), int(y + ch - 9), 6, 6)

            if label == self.current_node:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor("#1a56db"), 3))
                painter.drawRect(int(x) + 1, int(y) + 1, int(cw) - 2, int(ch) - 2)

        painter.end()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class LightboxCalibrationApp(QMainWindow):
    def __init__(self, simulate: bool = False):
        super().__init__()
        self.simulate = simulate

        # Campaign state
        self.campaign_dir: Path | None = None
        self.config: dict | None = None
        self.grid: cm.GridConfig = cm.GridConfig()
        self.captures_csv: Path | None = None
        self.captures_cache: list[dict] = []
        self.cursor: dict | None = None
        self.analysis: dict | None = None
        self._manual_covered_nodes: set[str] = set()
        self._pick_mode = "reference"

        self.current_level_index = 0
        self.current_level_variac = 0.0
        self.current_level_started_at = 0.0
        self.pass_start_ts = 0.0
        self._last_cursor_key = None

        self.warmup_active = False
        self.warmup_remaining = 0.0
        self.paused = False
        self.capture_in_progress = False
        self.capture_samples: list[float] = []
        self.capture_start_ts = 0.0
        self.last_capture_id: str | None = None
        self._undo_snapshot: dict | None = None
        self._last_active_quantity: str = "lux"

        # Sensor state -- same classes used by low_light_app, whether fed by
        # real hardware threads or (in --simulate) a QTimer-driven synthetic
        # field. Everything downstream (live readouts, timeseries plot,
        # capture sampling) is identical either way.
        self.lux_live = sensors.LiveValue()
        self.lux_history = sensors.TimeSeriesBuffer(max_age_s=sensors.TIMESERIES_HISTORY_S)
        self.irr_store = sensors.HistoryIrradiance(window_seconds=sensors.IRR_WINDOW_S, history_seconds=sensors.TIMESERIES_HISTORY_S)
        self._lux_stop = threading.Event()
        self._lux_thread = None
        self._irr_thread = None
        self._sensor_connect_status_pending = False

        self._build_ui()

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(150)
        self._live_timer.timeout.connect(self._guard(self._update_live_readouts))
        self._live_timer.start()

        self._ts_redraw_timer = QTimer(self)
        self._ts_redraw_timer.setInterval(sensors.TIMESERIES_REDRAW_MS)
        self._ts_redraw_timer.timeout.connect(self._guard(self._redraw_timeseries))
        self._ts_redraw_timer.start()

        if self.simulate:
            self._sim_timer = QTimer(self)
            self._sim_timer.setInterval(150)
            self._sim_timer.timeout.connect(self._guard(self._sim_tick))
            self._sim_timer.start()
            self.set_status("Simulation mode -- synthetic light field, no hardware needed.", "info")
        else:
            self._auto_connect_sensors_if_needed()

    # ------------------------------------------------------------------
    # Error-guarding (same rationale as low_light_app._guard: an unhandled
    # exception in a Qt slot aborts the whole process with no traceback).
    # ------------------------------------------------------------------

    def _guard(self, fn):
        def _wrapped(*args):
            try:
                return fn()
            except Exception as e:
                _log("error", "unhandled_slot_exception", slot=getattr(fn, "__name__", str(fn)), error=str(e))
                try:
                    self.set_status(f"Internal error (caught): {e}", "error")
                except Exception:
                    pass
        return _wrapped

    def set_status(self, text: str, stage: str = "info") -> None:
        self.status_badge.setText(text)
        self.status_badge.setStyleSheet(_STAGE_STYLE.get(stage, _STAGE_STYLE["info"]))

    def closeEvent(self, event) -> None:
        if self.capture_in_progress:
            resp = QMessageBox.question(
                self, "Capture in progress",
                "A capture is currently sampling. Closing now will abandon it (already-saved "
                "captures are safe on disk). Close anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        event.accept()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        title = "Lightbox Spatial Calibration" + (" [SIMULATE]" if self.simulate else "")
        self.setWindowTitle(title)
        self.setStyleSheet(_STYLESHEET)
        self.resize(1280, 860)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet("background:#1a56db; border-bottom:2px solid #1246c0;")
        hlay = QHBoxLayout(header)
        hlay.setContentsMargins(14, 0, 14, 0)
        title_lbl = QLabel(title.upper())
        title_lbl.setStyleSheet("color:#ffffff; font-size:14pt; font-weight:bold;")
        hlay.addWidget(title_lbl)
        hlay.addStretch()
        about_btn = QPushButton("About / Limitations")
        about_btn.setStyleSheet(
            "QPushButton { background:#ffffff; color:#1a56db; border:2px solid #ffffff; "
            "border-radius:6px; font-size:10pt; font-weight:bold; min-height:0; padding:4px 10px; }"
        )
        about_btn.clicked.connect(self._guard(self._show_about))
        hlay.addWidget(about_btn)
        root.addWidget(header)

        self.status_badge = QLabel("Ready -- create or resume a campaign in Setup")
        self.status_badge.setWordWrap(True)
        self.status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_badge.setStyleSheet(_STAGE_STYLE["info"])
        root.addWidget(self.status_badge)

        self._tab_widget = QTabWidget()
        root.addWidget(self._tab_widget, 1)

        self.setup_tab = self._build_setup_tab()
        self._tab_widget.addTab(self.setup_tab, "Setup")

        self.sensors_tab = self._build_sensors_tab()
        self._tab_widget.addTab(self.sensors_tab, "Sensors")

        self.run_tab = self._build_run_tab()
        self._tab_widget.addTab(self.run_tab, "Run")

        self.analysis_tab = self._build_analysis_tab()
        self._tab_widget.addTab(self.analysis_tab, "Analysis")

        QShortcut(QKeySequence("Space"), self, activated=self._guard(self._on_capture_clicked))
        QShortcut(QKeySequence("Return"), self, activated=self._guard(self._on_capture_clicked))
        QShortcut(QKeySequence("Enter"), self, activated=self._guard(self._on_capture_clicked))
        QShortcut(QKeySequence("F"), self, activated=self._guard(self._on_capture_clicked))

    def _show_about(self) -> None:
        QMessageBox.information(
            self, "About / Known limitations",
            "Lightbox Spatial Calibration Tool\n\n"
            "Known limitations (see docs/lightbox-calibration-plan.md section 9):\n\n"
            "- Halogen spectrum shifts with variac voltage: a lux-to-irradiance "
            "conversion derived at one level is invalid at another.\n"
            "- Neither sensor sees the solar spectrum: this corrects for position "
            "within the box only, not box-vs-AM1.5.\n"
            "- Two passes means two lamp thermal states: always run LUX and IRR "
            "back to back at a level.\n"
            "- Sensor head height matters at this distance from the lamps: keep "
            "the jig geometry fixed between calibration and real runs.\n"
            "- K assumes the panel sits exactly where the covered nodes are: mark "
            "the panel outline on the acrylic.\n"
            "- 50mm grid pitch cannot resolve structure below ~100mm.",
        )

    # ------------------------------------------------------------------
    # Setup tab
    # ------------------------------------------------------------------

    def _build_setup_tab(self) -> QWidget:
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        content = QWidget()
        outer.setWidget(content)
        layout = QHBoxLayout(content)

        form_col = QWidget()
        form = QVBoxLayout(form_col)
        form.setSpacing(10)

        campaign_box = QGroupBox("Campaign")
        cform = QFormLayout(campaign_box)
        self.campaign_name_edit = QLineEdit("lightbox")
        self.operator_edit = QLineEdit()
        self.output_folder_edit = QLineEdit(str(DEFAULT_CALIBRATION_DIR))
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._guard(self._browse_output_folder))
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_folder_edit)
        out_row.addWidget(browse_btn)
        cform.addRow("Campaign name / slug", self.campaign_name_edit)
        cform.addRow("Operator", self.operator_edit)
        cform.addRow("Output folder", out_row)
        form.addWidget(campaign_box)

        grid_box = QGroupBox("Grid")
        gform = QFormLayout(grid_box)
        self.cols_edit = QLineEdit(str(cm.DEFAULT_COLS))
        self.rows_edit = QLineEdit(str(cm.DEFAULT_ROWS))
        self.pitch_edit = QLineEdit(str(cm.DEFAULT_PITCH_MM))
        for edit in (self.cols_edit, self.rows_edit, self.pitch_edit):
            edit.textChanged.connect(self._guard(self._on_grid_fields_changed))
        gform.addRow("Columns", self.cols_edit)
        gform.addRow("Rows", self.rows_edit)
        gform.addRow("Pitch (mm)", self.pitch_edit)
        form.addWidget(grid_box)

        timing_box = QGroupBox("Timing")
        tform = QFormLayout(timing_box)
        self.dwell_edit = QLineEdit(str(cm.DEFAULT_DWELL_SECONDS))
        self.warmup_edit = QLineEdit(str(cm.DEFAULT_WARMUP_SECONDS))
        self.drift_flag_edit = QLineEdit(str(cm.DEFAULT_DRIFT_FLAG_PCT))
        tform.addRow("Dwell (s)", self.dwell_edit)
        tform.addRow("Warm-up (s)", self.warmup_edit)
        tform.addRow("Drift flag (%)", self.drift_flag_edit)
        form.addWidget(timing_box)

        panel_box = QGroupBox("Panel footprint")
        pform = QFormLayout(panel_box)
        self.panel_name_edit = QLineEdit()
        self.panel_width_edit = QLineEdit()
        self.panel_height_edit = QLineEdit()
        for edit in (self.panel_width_edit, self.panel_height_edit):
            edit.textChanged.connect(self._guard(self._on_grid_fields_changed))
        self.manual_pick_checkbox = QCheckBox("Pick nodes manually (click grid below)")
        self.manual_pick_checkbox.stateChanged.connect(self._guard(self._on_grid_fields_changed))
        pform.addRow("Panel name", self.panel_name_edit)
        pform.addRow("Width (mm)", self.panel_width_edit)
        pform.addRow("Height (mm)", self.panel_height_edit)
        pform.addRow(self.manual_pick_checkbox)
        self.footprint_warning_lbl = QLabel("")
        self.footprint_warning_lbl.setWordWrap(True)
        self.footprint_warning_lbl.setStyleSheet("color:#8a5300; font-size:9pt;")
        pform.addRow(self.footprint_warning_lbl)
        form.addWidget(panel_box)

        node_box = QGroupBox("Reference & sensor nodes")
        nform = QFormLayout(node_box)
        self.reference_node_edit = QLineEdit(cm.DEFAULT_REFERENCE_NODE)
        self.sensor_node_edit = QLineEdit(cm.DEFAULT_REFERENCE_NODE)
        for edit in (self.reference_node_edit, self.sensor_node_edit):
            edit.textChanged.connect(self._guard(self._on_grid_fields_changed))
        self.pick_mode_combo = QComboBox()
        self.pick_mode_combo.addItems(["Reference node", "Sensor node", "Panel footprint (manual)"])
        self.pick_mode_combo.currentIndexChanged.connect(self._guard(self._on_pick_mode_changed))
        nform.addRow("Reference node", self.reference_node_edit)
        nform.addRow("Sensor node", self.sensor_node_edit)
        nform.addRow("Click grid sets", self.pick_mode_combo)
        form.addWidget(node_box)

        misc_box = QGroupBox("Other")
        mform = QFormLayout(misc_box)
        self.sensor_height_edit = QLineEdit("12")
        self.lamp_note_edit = QLineEdit("halogen MR16 array, diffuser in place")
        self.campaign_note_edit = QTextEdit()
        self.campaign_note_edit.setMaximumHeight(60)
        mform.addRow("Sensor head height (mm)", self.sensor_height_edit)
        mform.addRow("Lamp note", self.lamp_note_edit)
        mform.addRow("Campaign note", self.campaign_note_edit)
        form.addWidget(misc_box)

        btn_row = QHBoxLayout()
        new_btn = QPushButton("New Campaign")
        new_btn.setProperty("primary", True)
        new_btn.clicked.connect(self._guard(self._on_new_campaign))
        resume_btn = QPushButton("Resume Campaign…")
        resume_btn.clicked.connect(self._guard(self._on_resume_campaign))
        btn_row.addWidget(new_btn)
        btn_row.addWidget(resume_btn)
        form.addLayout(btn_row)
        form.addStretch()

        layout.addWidget(form_col, 1)

        preview_col = QVBoxLayout()
        preview_col.addWidget(QLabel("Preview (green = panel footprint, blue dot = reference, red square = sensor):"))
        self.setup_grid = GridWidget(self.grid, clickable=True)
        self.setup_grid.on_node_clicked = self._on_setup_grid_clicked
        preview_col.addWidget(self.setup_grid, 1)
        preview_wrap = QWidget()
        preview_wrap.setLayout(preview_col)
        layout.addWidget(preview_wrap, 1)

        self._on_grid_fields_changed()
        return outer

    def _browse_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Output folder", self.output_folder_edit.text())
        if folder:
            self.output_folder_edit.setText(folder)

    def _current_setup_grid(self) -> cm.GridConfig | None:
        try:
            return cm.GridConfig(
                cols=int(self.cols_edit.text()), rows=int(self.rows_edit.text()),
                pitch_mm=float(self.pitch_edit.text()),
            )
        except (ValueError, TypeError):
            return None

    def _on_grid_fields_changed(self) -> None:
        grid = self._current_setup_grid()
        if grid is None:
            return
        self.setup_grid.set_grid(grid)

        if self.manual_pick_checkbox.isChecked():
            covered = self._manual_covered_nodes & set(cm.all_node_labels(grid))
        else:
            try:
                width = float(self.panel_width_edit.text())
                height = float(self.panel_height_edit.text())
                covered = set(cm.covered_nodes(grid, width, height))
            except (ValueError, TypeError):
                covered = set()

        self.setup_grid.covered_nodes = covered
        warning = cm.footprint_warning(list(covered)) if covered else None
        self.footprint_warning_lbl.setText(warning or "")

        self.setup_grid.reference_node = self.reference_node_edit.text().strip().upper() or None
        self.setup_grid.sensor_node = self.sensor_node_edit.text().strip().upper() or None
        self.setup_grid.update()

    def _on_pick_mode_changed(self) -> None:
        self._pick_mode = ["reference", "sensor", "panel"][self.pick_mode_combo.currentIndex()]

    def _on_setup_grid_clicked(self, label: str) -> None:
        if self._pick_mode == "reference":
            self.reference_node_edit.setText(label)
        elif self._pick_mode == "sensor":
            self.sensor_node_edit.setText(label)
        else:
            if not self.manual_pick_checkbox.isChecked():
                self.manual_pick_checkbox.setChecked(True)
            if label in self._manual_covered_nodes:
                self._manual_covered_nodes.discard(label)
            else:
                self._manual_covered_nodes.add(label)
        self._on_grid_fields_changed()

    def _on_new_campaign(self) -> None:
        grid = self._current_setup_grid()
        if grid is None:
            QMessageBox.warning(self, "Invalid grid", "Columns/rows/pitch must be valid numbers.")
            return
        try:
            dwell = float(self.dwell_edit.text())
            warmup = float(self.warmup_edit.text())
            drift_flag = float(self.drift_flag_edit.text())
            sensor_height = float(self.sensor_height_edit.text() or 0)
            width = float(self.panel_width_edit.text() or 0)
            height = float(self.panel_height_edit.text() or 0)
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "Timing/panel fields must be valid numbers.")
            return

        reference_node = self.reference_node_edit.text().strip().upper() or cm.DEFAULT_REFERENCE_NODE
        sensor_node = self.sensor_node_edit.text().strip().upper() or reference_node
        try:
            cm.parse_node_label(reference_node)
            cm.parse_node_label(sensor_node)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid node", str(e))
            return

        manual_covered = sorted(self._manual_covered_nodes) if self.manual_pick_checkbox.isChecked() else None
        covered = manual_covered if manual_covered is not None else cm.covered_nodes(grid, width, height)
        panel = {
            "name": self.panel_name_edit.text().strip(), "width_mm": width, "height_mm": height,
            "anchor": "centered" if manual_covered is None else "manual", "covered_nodes": covered,
        }

        name = self.campaign_name_edit.text().strip() or "lightbox"
        operator = self.operator_edit.text().strip() or "unknown"
        output_folder = Path(self.output_folder_edit.text().strip() or str(DEFAULT_CALIBRATION_DIR))
        campaign_id = cm.new_campaign_id(name)
        campaign_dir = output_folder / campaign_id

        config = cm.default_campaign_config(
            campaign_id=campaign_id, operator=operator, grid=grid, reference_node=reference_node,
            dwell_seconds=dwell, warmup_seconds=warmup, drift_flag_pct=drift_flag, panel=panel,
            sensor_node=sensor_node, sensor_head_height_mm=sensor_height,
            lamp_note=self.lamp_note_edit.text().strip(),
        )
        if self.campaign_note_edit.toPlainText().strip():
            config["note"] = self.campaign_note_edit.toPlainText().strip()

        cm.save_campaign_config(campaign_dir, config)
        self._load_campaign(campaign_dir, config)
        self.set_status(f"Campaign {campaign_id} created.", "success")
        self._tab_widget.setCurrentWidget(self.run_tab)
        self._prompt_new_level()

    def _on_resume_campaign(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Resume campaign", str(DEFAULT_CALIBRATION_DIR))
        if not folder:
            return
        campaign_dir = Path(folder)
        try:
            config = cm.load_campaign_config(campaign_dir)
        except Exception as e:
            QMessageBox.warning(self, "Could not resume", f"Failed to read campaign.json: {e}")
            return

        self._load_campaign(campaign_dir, config)
        levels = config.get("levels", [])
        if levels:
            last = levels[-1]
            self.current_level_index = last["level_index"]
            self.current_level_variac = last["variac_vac"]
        self.cursor = cm.next_cursor(
            self.captures_cache, self.grid, config["reference_node"],
            current_level_index=self.current_level_index or None,
        )
        self._tab_widget.setCurrentWidget(self.run_tab)

        if self.cursor is None:
            self.set_status(f"Resumed {config['campaign_id']} -- level complete.", "success")
            self._prompt_end_of_level()
        else:
            self.pass_start_ts = time.time()
            self._update_header()
            self._highlight_current_node()
            self.set_status(
                f"Resumed {config['campaign_id']} at level {self.cursor['level_index']} "
                f"{self.cursor['pass']} pass, node {self.cursor['node_label']}.", "success",
            )

    def _load_campaign(self, campaign_dir: Path, config: dict) -> None:
        self.campaign_dir = campaign_dir
        self.config = config
        self.grid = cm.grid_from_campaign(config)
        self.captures_csv = cm.captures_csv_path(campaign_dir)
        self.captures_cache = cm.read_captures(self.captures_csv)
        self._undo_snapshot = None
        self.undo_redo_btn.setEnabled(False)
        self.run_grid.set_grid(self.grid)
        self.setup_grid.set_grid(self.grid)
        self.run_grid.reference_node = config["reference_node"]
        self.run_grid.sensor_node = config["sensor_node"]
        self.run_grid.covered_nodes = set(cm.effective_covered_nodes(config, self.grid))
        self._rebuild_run_grid()

    # ------------------------------------------------------------------
    # Sensors tab
    # ------------------------------------------------------------------

    def _build_sensors_tab(self) -> QWidget:
        outer = QWidget()
        layout = QVBoxLayout(outer)

        if self.simulate:
            note = QLabel("Simulation mode: lux/irradiance are synthetic. No COM ports needed.")
            note.setStyleSheet("color:#1b5e20; font-weight:bold;")
            layout.addWidget(note)
        else:
            port_box = QGroupBox("Ports")
            pform = QFormLayout(port_box)
            self.lux_port_combo = QComboBox()
            self.irr_port_combo = QComboBox()
            pform.addRow("Lux meter (LT68)", self.lux_port_combo)
            pform.addRow("Irradiance sensor", self.irr_port_combo)
            layout.addWidget(port_box)

            btn_row = QHBoxLayout()
            refresh_btn = QPushButton("Refresh Ports")
            refresh_btn.clicked.connect(self._guard(self._refresh_port_lists))
            connect_btn = QPushButton("Connect")
            connect_btn.setProperty("primary", True)
            connect_btn.clicked.connect(self._guard(self._manual_connect_sensors))
            btn_row.addWidget(refresh_btn)
            btn_row.addWidget(connect_btn)
            layout.addLayout(btn_row)

        live_row = QHBoxLayout()
        self.lux_value_lbl = QLabel("-- lx")
        self.lux_value_lbl.setStyleSheet("font-size:20pt; font-weight:bold;")
        self.irr_value_lbl = QLabel("-- W/m²")
        self.irr_value_lbl.setStyleSheet("font-size:20pt; font-weight:bold;")
        self.lux_badge = QLabel("● Lux: --")
        self.irr_badge = QLabel("● Irr: --")
        live_row.addWidget(self.lux_value_lbl)
        live_row.addWidget(self.lux_badge)
        live_row.addSpacing(30)
        live_row.addWidget(self.irr_value_lbl)
        live_row.addWidget(self.irr_badge)
        live_row.addStretch()
        layout.addLayout(live_row)

        ts_fig = Figure(dpi=100, facecolor="#f0f2f5")
        self.ts_canvas = FigureCanvas(ts_fig)
        self.lux_ts_ax = ts_fig.add_subplot(1, 2, 1)
        self.irr_ts_ax = ts_fig.add_subplot(1, 2, 2)
        self._style_timeseries_axes()
        ts_fig.tight_layout()
        layout.addWidget(self.ts_canvas, 1)

        if not self.simulate:
            self._refresh_port_lists()

        return outer

    def _style_timeseries_axes(self) -> None:
        self.lux_ts_ax.set_title("Lux (live)", fontsize=9)
        self.lux_ts_ax.set_xlabel("Seconds ago", fontsize=8)
        self.lux_ts_ax.grid(True, alpha=0.3)
        self.irr_ts_ax.set_title("Irradiance (live)", fontsize=9)
        self.irr_ts_ax.set_xlabel("Seconds ago", fontsize=8)
        self.irr_ts_ax.grid(True, alpha=0.3)

    def _refresh_port_lists(self) -> dict:
        ports = list(serial.tools.list_ports.comports())
        detected = sensors.detect_sensor_ports()
        cached_lux = sensors.load_cached_port("lux")
        cached_irr = sensors.load_cached_port("irr")

        for combo, cached, detected_port in (
            (self.lux_port_combo, cached_lux, detected["lux"]),
            (self.irr_port_combo, cached_irr, detected["irr"]),
        ):
            combo.blockSignals(True)
            combo.clear()
            seen = set()
            for p in ports:
                label = f"{p.device} — {p.description}" if p.description else p.device
                combo.addItem(label, p.device)
                seen.add(p.device)
            if cached and cached not in seen:
                combo.addItem(f"{cached} (not currently connected)", cached)
            preferred = detected_port or cached
            if preferred:
                idx = combo.findData(preferred)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        return detected

    def _auto_connect_sensors_if_needed(self) -> None:
        detected = self._refresh_port_lists()
        lux_running = self._lux_thread is not None and self._lux_thread.is_alive()
        irr_running = self._irr_thread is not None and self._irr_thread.is_alive()
        if (detected["lux"] and not lux_running) or (detected["irr"] and not irr_running):
            self.set_status("Auto-detected sensor(s) -- connecting…", "info")
            self._connect_sensors()
        elif not detected["lux"] and not detected["irr"]:
            self.set_status(
                "Could not auto-detect the lux meter (CP210x) or irradiance sensor (CH9102) -- "
                "check they're plugged in, or pick their ports manually in the Sensors tab.",
                "warning",
            )

    def _manual_connect_sensors(self) -> None:
        self.set_status("Connecting sensor(s)…", "info")
        self._connect_sensors()

    def _connect_sensors(self) -> None:
        lux_port = self.lux_port_combo.currentData() or self.lux_port_combo.currentText().strip()
        irr_port = self.irr_port_combo.currentData() or self.irr_port_combo.currentText().strip()
        started_any = False

        if lux_port:
            if self._lux_thread is not None and self._lux_thread.is_alive():
                pass
            elif sensors.port_in_use(lux_port):
                self.set_status(f"{lux_port} is in use -- close the Low Light Testing app first.", "error")
            else:
                sensors.save_cached_port("lux", lux_port)
                self._lux_stop.clear()
                self._lux_thread = threading.Thread(
                    target=sensors.lux_poll_loop, args=(lux_port, self.lux_live, self._lux_stop, self.lux_history),
                    daemon=True,
                )
                self._lux_thread.start()
                started_any = True

        if irr_port:
            if self._irr_thread is not None and self._irr_thread.is_alive():
                pass
            elif sensors.port_in_use(irr_port):
                self.set_status(f"{irr_port} is in use -- close the Low Light Testing app first.", "error")
            else:
                sensors.save_cached_port("irr", irr_port)
                from listen import serial_reader
                self._irr_thread = threading.Thread(
                    target=serial_reader,
                    args=(irr_port, sensors.IRR_BAUD_DEFAULT, self.irr_store, 10**9, 1.0, 2.0, 30.0, _log),
                    daemon=True,
                )
                self._irr_thread.start()
                started_any = True

        if started_any:
            self._sensor_connect_status_pending = True
        if not lux_port and not irr_port:
            self.set_status("Pick at least one sensor COM port before connecting.", "warning")

    def _update_live_readouts(self) -> None:
        now = time.time()
        lux_value, lux_ts, lux_err = self.lux_live.snapshot()
        lux_is_live = lux_value is not None and (lux_ts is None or now - lux_ts <= sensors.LIVE_STALE_S)
        if lux_is_live:
            self.lux_value_lbl.setText(f"{lux_value:.1f} lx")
            self.lux_badge.setText("● Lux: live")
            self.lux_badge.setStyleSheet("color:#1b5e20; font-weight:bold;")
        elif lux_err:
            self.lux_value_lbl.setText("-- lx (error)")
            self.lux_badge.setText("● Lux: error")
            self.lux_badge.setStyleSheet("color:#9b1c1c; font-weight:bold;")
        else:
            self.lux_value_lbl.setText("-- lx")
            self.lux_badge.setText("● Lux: --")
            self.lux_badge.setStyleSheet("color:#8a5300; font-weight:bold;")

        irr_value, irr_ts = self.irr_store.latest()
        irr_is_live = irr_value is not None and (irr_ts is None or now - irr_ts <= sensors.LIVE_STALE_S)
        if irr_is_live:
            self.irr_value_lbl.setText(f"{irr_value:.4f} W/m²")
            self.irr_badge.setText("● Irr: live")
            self.irr_badge.setStyleSheet("color:#1b5e20; font-weight:bold;")
        else:
            self.irr_value_lbl.setText("-- W/m²")
            self.irr_badge.setText("● Irr: --")
            self.irr_badge.setStyleSheet("color:#8a5300; font-weight:bold;")

        if self._sensor_connect_status_pending and not self.simulate:
            lux_wanted = bool(self.lux_port_combo.currentData())
            irr_wanted = bool(self.irr_port_combo.currentData())
            if ((not lux_wanted) or lux_is_live) and ((not irr_wanted) or irr_is_live):
                self._sensor_connect_status_pending = False
                self.set_status("Sensors connected.", "success")

        # Run tab's live capture readout mirrors whichever quantity is active.
        # Tracked in self._last_active_quantity (not read straight off
        # self.cursor) because the cursor is None for as long as an
        # end-of-pass/end-of-level dialog is open -- gating on it directly
        # froze this readout for the whole time that modal was up, which is
        # exactly when the operator most wants to keep watching the sensor
        # settle. Qt's own timers keep firing under a modal dialog, so this
        # was purely the gate, not the underlying poll/timer being stalled.
        if self.cursor is not None:
            self._last_active_quantity = self.cursor["pass"]
        active_val = lux_value if self._last_active_quantity == "lux" else irr_value
        unit = "lx" if self._last_active_quantity == "lux" else "W/m²"
        self.live_readout_lbl.setText(f"{active_val:.2f} {unit}" if active_val is not None else "-- ")

    def _redraw_timeseries(self) -> None:
        now = time.time()
        lux_times, lux_values = self.lux_history.snapshot()
        irr_times, irr_values = self.irr_store.history.snapshot()
        self.lux_ts_ax.clear()
        self.irr_ts_ax.clear()
        if lux_times:
            self.lux_ts_ax.plot([t - now for t in lux_times], lux_values, color="#1a56db", linewidth=1.2)
        if irr_times:
            self.irr_ts_ax.plot([t - now for t in irr_times], irr_values, color="#c53030", linewidth=1.2)
        self._style_timeseries_axes()
        self.ts_canvas.figure.tight_layout()
        self.ts_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Simulation driver
    # ------------------------------------------------------------------

    def _sim_tick(self) -> None:
        if self.config is None:
            node_label = cm.DEFAULT_REFERENCE_NODE
        elif self.cursor is not None:
            node_label = self.cursor["node_label"]
        else:
            node_label = self.config["reference_node"]

        cursor_key = (self.cursor["level_index"], self.cursor["pass"]) if self.cursor else None
        if cursor_key != self._last_cursor_key:
            self.pass_start_ts = time.time()
            self._last_cursor_key = cursor_key

        try:
            x_mm, y_mm = cm.label_to_coords_mm(node_label, self.grid.pitch_mm)
        except ValueError:
            x_mm, y_mm = 0.0, 0.0

        vac = self.current_level_variac or 50.0
        drift_fraction = min(1.0, (time.time() - self.pass_start_ts) / 300.0)

        lux_val = simulate_field_value(x_mm, y_mm, self.grid, vac, "lux", drift_fraction)
        irr_val = simulate_field_value(x_mm, y_mm, self.grid, vac, "irr", drift_fraction)
        self.lux_live.set(lux_val)
        self.lux_history.add(lux_val)
        self.irr_store.add(irr_val)

    # ------------------------------------------------------------------
    # Run tab
    # ------------------------------------------------------------------

    def _build_run_tab(self) -> QWidget:
        outer = QWidget()
        layout = QVBoxLayout(outer)

        self.header_lbl = QLabel("No campaign loaded")
        self.header_lbl.setStyleSheet("font-size:12pt; font-weight:bold; padding:6px;")
        layout.addWidget(self.header_lbl)

        self.warmup_row = QWidget()
        wlay = QHBoxLayout(self.warmup_row)
        self.warmup_lbl = QLabel("")
        skip_warmup_btn = QPushButton("Skip warm-up")
        skip_warmup_btn.clicked.connect(self._guard(self._skip_warmup))
        wlay.addWidget(self.warmup_lbl, 1)
        wlay.addWidget(skip_warmup_btn)
        self.warmup_row.hide()
        layout.addWidget(self.warmup_row)

        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split, 1)

        grid_col = QVBoxLayout()
        grid_hint = QLabel("Right-click a captured node to redo it -- capture continues from where you left off.")
        grid_hint.setStyleSheet("color:#555; font-size:9pt;")
        grid_col.addWidget(grid_hint)
        self.run_grid = GridWidget(self.grid, clickable=True)
        self.run_grid.on_node_right_clicked = self._on_redo_node_clicked
        grid_col.addWidget(self.run_grid, 1)
        grid_wrap = QWidget()
        grid_wrap.setLayout(grid_col)
        split.addWidget(grid_wrap)

        panel = QWidget()
        play = QVBoxLayout(panel)

        self.live_readout_lbl = QLabel("--")
        self.live_readout_lbl.setStyleSheet("font-size:28pt; font-weight:bold;")
        self.live_readout_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        play.addWidget(self.live_readout_lbl)

        self.capture_btn = QPushButton("Capture  (Space / Enter / F)")
        self.capture_btn.setProperty("primary", True)
        self.capture_btn.clicked.connect(self._guard(self._on_capture_clicked))
        self.capture_btn.setEnabled(False)
        play.addWidget(self.capture_btn)

        self.capture_progress = QProgressBar()
        self.capture_progress.setRange(0, 100)
        play.addWidget(self.capture_progress)

        self.result_card = QLabel("")
        self.result_card.setWordWrap(True)
        self.result_card.setStyleSheet(_STAGE_STYLE["info"])
        play.addWidget(self.result_card)

        btn_row1 = QHBoxLayout()
        redo_btn = QPushButton("Redo Last")
        redo_btn.clicked.connect(self._guard(self._redo_last))
        skip_btn = QPushButton("Skip Node")
        skip_btn.clicked.connect(self._guard(self._skip_node))
        btn_row1.addWidget(redo_btn)
        btn_row1.addWidget(skip_btn)
        play.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.pause_btn = QPushButton("Pause Pass")
        self.pause_btn.clicked.connect(self._guard(self._toggle_pause))
        end_pass_btn = QPushButton("End Pass Early")
        end_pass_btn.setProperty("danger", True)
        end_pass_btn.clicked.connect(self._guard(self._end_pass_early))
        btn_row2.addWidget(self.pause_btn)
        btn_row2.addWidget(end_pass_btn)
        play.addLayout(btn_row2)

        self.undo_redo_btn = QPushButton("Undo Last Redo")
        self.undo_redo_btn.setToolTip(
            "Restores whatever was just redone (Redo Last, a right-click node redo, or "
            "Redo This Pass) -- only the single most recent redo can be undone."
        )
        self.undo_redo_btn.setEnabled(False)
        self.undo_redo_btn.clicked.connect(self._guard(self._undo_last_redo))
        play.addWidget(self.undo_redo_btn)

        play.addStretch()
        split.addWidget(panel)
        split.setSizes([700, 400])

        return outer

    def _rebuild_run_grid(self) -> None:
        self.run_grid.reset_cells()
        if not self.captures_cache or self.cursor is None:
            level_index = self.current_level_index
            quantity = "lux"
        else:
            level_index, quantity = self.cursor["level_index"], self.cursor["pass"]

        active = cm.active_captures(self.captures_cache)
        all_ids = {c["capture_id"] for c in cm.active_captures(self.captures_cache)}
        superseded_nodes = {
            c["node_label"] for c in self.captures_cache
            if c.get("superseded") and c["level_index"] == level_index and c["quantity"] == quantity
        }
        for c in active:
            if c["level_index"] != level_index or c["quantity"] != quantity or c["capture_kind"] != "grid":
                continue
            self._paint_capture_cell(c)
        self.run_grid.redone_nodes = superseded_nodes
        self.run_grid.update()

    def _paint_capture_cell(self, record: dict) -> None:
        label = record["node_label"]
        if record.get("note") in ("skipped", "pass_ended_early"):
            state = "skipped"
        elif record.get("three_sigma_pct") is not None and record["three_sigma_pct"] > AMBER_3SIGMA_PCT:
            state = "flagged"
        else:
            state = "captured"
        self.run_grid.cell_state[label] = state
        self.run_grid.cell_value[label] = record.get("mean")
        if record.get("mean") is not None:
            stdev = record.get("stdev")
            pct = record.get("three_sigma_pct")
            self.run_grid.hover_text[label] = (
                f"{label}: {record['mean']:.2f} ± {stdev:.2f} (3σ/avg {pct:.1f}%, n={record['n_samples']})"
                if stdev is not None else f"{label}: {record['mean']:.2f} (n={record['n_samples']})"
            )
        else:
            self.run_grid.hover_text[label] = f"{label}: skipped"

    def _highlight_current_node(self) -> None:
        self.run_grid.current_node = self.cursor["node_label"] if self.cursor else None
        self.run_grid.update()

    def _update_header(self) -> None:
        if self.config is None or self.cursor is None:
            self.header_lbl.setText("No active pass")
            self.capture_btn.setEnabled(False)
            return
        node_count = self.grid.node_count
        idx = self.cursor["sequence_index"]
        progress = "ref" if idx < 0 else f"{idx + 1} / {node_count}"
        elapsed = time.time() - self.current_level_started_at
        mm, ss = divmod(int(elapsed), 60)
        self.header_lbl.setText(
            f"Level {self.cursor['level_index']}   |   {self.current_level_variac:.1f} VAC   |   "
            f"{self.cursor['pass'].upper()} pass   |   Node {self.cursor['node_label']}   |   "
            f"{progress}   |   elapsed {mm:02d}:{ss:02d}"
        )
        self.capture_btn.setEnabled(not self.warmup_active and not self.paused and not self.capture_in_progress)
        self._highlight_current_node()

    # -- Level / warm-up flow --------------------------------------------------

    def _prompt_new_level(self) -> None:
        level_index = len(self.config["levels"]) + 1
        vac, ok = QInputDialog.getDouble(
            self, "Variac level", f"Enter VAC for level {level_index}:", 0.0, 0.0, 130.0, 1,
        )
        if not ok:
            self.set_status("Level entry cancelled.", "warning")
            return

        self.current_level_index = level_index
        self.current_level_variac = vac
        self.current_level_started_at = time.time()
        self.config["levels"].append({
            "level_index": level_index, "variac_vac": vac, "started_at": cm._local_iso_now(),
            "passes": {"lux": "pending", "irr": "pending"},
        })
        cm.save_campaign_config(self.campaign_dir, self.config)
        self.cursor = cm.next_cursor(
            self.captures_cache, self.grid, self.config["reference_node"],
            current_level_index=self.current_level_index,
        )
        self._rebuild_run_grid()
        self._start_warmup()

    def _start_warmup(self) -> None:
        self.warmup_active = True
        self.warmup_remaining = self.config["warmup_seconds"]
        self.warmup_row.show()
        self.capture_btn.setEnabled(False)
        self._warmup_timer = QTimer(self)
        self._warmup_timer.setInterval(1000)
        self._warmup_timer.timeout.connect(self._guard(self._warmup_tick))
        self._warmup_timer.start()
        self._warmup_tick(initial=True)

    def _warmup_tick(self, initial: bool = False) -> None:
        if not initial:
            self.warmup_remaining -= 1
        self.warmup_lbl.setText(f"Warm-up: {max(0, int(self.warmup_remaining))}s remaining")
        if self.warmup_remaining <= 0:
            self._end_warmup()

    def _skip_warmup(self) -> None:
        if self.config["levels"]:
            self.config["levels"][-1]["warmup_skipped"] = 1
            cm.save_campaign_config(self.campaign_dir, self.config)
        self._end_warmup()

    def _end_warmup(self) -> None:
        if hasattr(self, "_warmup_timer"):
            self._warmup_timer.stop()
        self.warmup_active = False
        self.warmup_row.hide()
        self.pass_start_ts = time.time()
        self.set_status("Warm-up complete -- capture the reference node to begin.", "success")
        self._update_header()

    # -- Capture -----------------------------------------------------------

    def _on_capture_clicked(self) -> None:
        if self._tab_widget.currentWidget() is not self.run_tab:
            return
        if self.capture_in_progress or self.warmup_active or self.paused or self.cursor is None:
            return
        self.capture_in_progress = True
        self.capture_btn.setEnabled(False)
        self.capture_start_ts = time.time()
        self.capture_samples = []
        self._capture_timer = QTimer(self)
        self._capture_timer.setInterval(int(CAPTURE_SAMPLE_INTERVAL_S * 1000))
        self._capture_timer.timeout.connect(self._guard(self._capture_tick))
        self._capture_timer.start()

    def _capture_tick(self) -> None:
        quantity = self.cursor["pass"]
        if quantity == "lux":
            value, _, _ = self.lux_live.snapshot()
        else:
            value, _ = self.irr_store.latest()
        if value is not None:
            self.capture_samples.append(value)

        dwell = self.config["dwell_seconds"]
        elapsed = time.time() - self.capture_start_ts
        self.capture_progress.setValue(int(min(1.0, elapsed / dwell) * 100))
        if elapsed >= dwell:
            self._capture_timer.stop()
            self._finish_capture()

    def _finish_capture(self) -> None:
        cursor = self.cursor
        duration = time.time() - self.capture_start_ts
        port = self.lux_port_combo.currentData() if (not self.simulate and cursor["pass"] == "lux") else (
            self.irr_port_combo.currentData() if not self.simulate else "SIMULATED"
        )
        record = cm.make_capture_record(
            campaign_id=self.config["campaign_id"], level_index=cursor["level_index"],
            variac_vac=self.current_level_variac, quantity=cursor["pass"], capture_kind=cursor["next_kind"],
            node_label_=cursor["node_label"], grid=self.grid, sequence_index=cursor["sequence_index"],
            samples=self.capture_samples, duration_s=duration, sensor_port=port or "",
        )
        cm.append_capture(self.captures_csv, record)
        self.last_capture_id = record["capture_id"]
        self.capture_in_progress = False
        self.capture_progress.setValue(0)
        self._refresh_after_capture(record)

    def _show_result_card(self, record: dict) -> None:
        if record.get("mean") is None:
            self.result_card.setText(f"{record['node_label']}: skipped")
            self.result_card.setStyleSheet(_STAGE_STYLE["warning"])
            return
        stdev = record.get("stdev")
        pct = record.get("three_sigma_pct")
        text = f"{record['node_label']} ({record['capture_kind']}): {record['mean']:.2f} ± {stdev or 0:.2f}"
        if pct is not None:
            text += f"  (3σ/avg {pct:.1f}%, n={record['n_samples']})"
        self.result_card.setText(text)
        stage = "warning" if (pct is not None and pct > AMBER_3SIGMA_PCT) else "success"
        self.result_card.setStyleSheet(_STAGE_STYLE[stage])

    def _refresh_after_capture(self, record: dict) -> None:
        self.captures_cache = cm.read_captures(self.captures_csv)
        if record["capture_kind"] == "grid":
            self._paint_capture_cell(record)
            self.run_grid.update()
        self._show_result_card(record)

        if record["capture_kind"] == "ref_end":
            self._handle_end_of_pass(record)
            return
        self._advance_cursor()

    def _advance_cursor(self) -> None:
        self.cursor = cm.next_cursor(
            self.captures_cache, self.grid, self.config["reference_node"],
            current_level_index=self.current_level_index,
        )
        if self.cursor is None:
            self._handle_end_of_level()
        else:
            if self.cursor["level_index"] != self.current_level_index or self._pass_changed():
                self._rebuild_run_grid()
            self._update_header()

    def _pass_changed(self) -> bool:
        key = (self.cursor["level_index"], self.cursor["pass"])
        changed = key != self._last_cursor_key
        self._last_cursor_key = key
        return changed

    def _handle_end_of_pass(self, record: dict) -> None:
        quantity = record["quantity"]
        level_index = record["level_index"]
        ref_start_row = next(
            (c for c in reversed(cm.active_captures(self.captures_cache))
             if c["level_index"] == level_index and c["quantity"] == quantity and c["capture_kind"] == "ref_start"),
            None,
        )
        ref_start_val = ref_start_row["mean"] if ref_start_row else None
        ref_end_val = record["mean"]
        d_pct = cm.drift_pct(ref_start_val, ref_end_val) if ref_start_val else None
        flagged = d_pct is not None and abs(d_pct) > self.config["drift_flag_pct"]

        msg = f"{quantity.upper()} pass complete.\nDrift: {d_pct:.2f}%" if d_pct is not None else f"{quantity.upper()} pass complete."
        if flagged:
            msg += "\n\nFLAGGED -- drift exceeds the configured threshold."

        box = QMessageBox(self)
        box.setWindowTitle("Pass complete")
        box.setText(msg)
        redo_btn = box.addButton("Redo this pass", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Continue", QMessageBox.ButtonRole.AcceptRole)
        box.exec()

        if box.clickedButton() is redo_btn:
            ids_to_supersede = [
                c["capture_id"] for c in cm.active_captures(self.captures_cache)
                if c["level_index"] == level_index and c["quantity"] == quantity
            ]
            confirm = QMessageBox.question(
                self, "Redo this pass?",
                f"This discards all {len(ids_to_supersede)} captures for this {quantity.upper()} "
                "pass (reference + grid nodes). The data stays on disk and Undo Last Redo can "
                "restore it, but only until your next redo action. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm == QMessageBox.StandardButton.Yes:
                self._record_undo_snapshot(level_index, quantity, ids_to_supersede)
                for capture_id in ids_to_supersede:
                    cm.mark_superseded(self.captures_csv, capture_id)
                self.captures_cache = cm.read_captures(self.captures_csv)
                self._rebuild_run_grid()

        self._advance_cursor()

    def _handle_end_of_level(self) -> None:
        if self.config["levels"]:
            self.config["levels"][-1]["passes"] = {"lux": "complete", "irr": "complete"}
            cm.save_campaign_config(self.campaign_dir, self.config)
        self._update_header()
        self._prompt_end_of_level()

    def _prompt_end_of_level(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Level complete")
        box.setText("Both passes are complete for this level.")
        add_btn = box.addButton("Add another level", QMessageBox.ButtonRole.AcceptRole)
        finish_btn = box.addButton("Finish campaign", QMessageBox.ButtonRole.DestructiveRole)
        box.exec()
        if box.clickedButton() is add_btn:
            self._prompt_new_level()
        elif box.clickedButton() is finish_btn:
            self._finish_campaign()

    def _finish_campaign(self) -> None:
        self.captures_cache = cm.read_captures(self.captures_csv)
        self.analysis = cm.analyze_campaign(self.captures_cache, self.config)
        self._populate_analysis_tab()
        self.set_status("Campaign complete -- analysis ready.", "success")
        self._tab_widget.setCurrentWidget(self.analysis_tab)

    # -- Secondary capture controls -----------------------------------------

    def _record_undo_snapshot(self, level_index: int, quantity: str, superseded_ids: list[str]) -> None:
        """Remember exactly what a redo action is about to supersede, plus
        which capture_ids already exist at this moment, so _undo_last_redo()
        can both restore the old rows and discard anything captured during
        the (mistaken) redo attempt itself. Must be called before the
        supersede loop runs, while self.captures_cache still reflects the
        pre-redo state. Only the single most recent redo is undoable -- a
        later redo action overwrites this snapshot."""
        self._undo_snapshot = {
            "level_index": level_index,
            "quantity": quantity,
            "superseded_ids": list(superseded_ids),
            "known_ids_before": {c["capture_id"] for c in self.captures_cache},
        }
        self.undo_redo_btn.setEnabled(True)

    def _undo_last_redo(self) -> None:
        snap = self._undo_snapshot
        if not snap:
            self.set_status("Nothing to undo.", "warning")
            return

        # Discard anything captured during the redo attempt being reversed --
        # otherwise it would sit alongside the restored originals as a second
        # active row for the same node.
        for c in cm.active_captures(self.captures_cache):
            if (
                c["level_index"] == snap["level_index"]
                and c["quantity"] == snap["quantity"]
                and c["capture_id"] not in snap["known_ids_before"]
            ):
                cm.mark_superseded(self.captures_csv, c["capture_id"])

        for capture_id in snap["superseded_ids"]:
            cm.mark_superseded(self.captures_csv, capture_id, superseded=False)

        self._undo_snapshot = None
        self.undo_redo_btn.setEnabled(False)
        self.captures_cache = cm.read_captures(self.captures_csv)
        self._rebuild_run_grid()
        self.cursor = cm.next_cursor(
            self.captures_cache, self.grid, self.config["reference_node"],
            current_level_index=self.current_level_index,
        )
        if self.cursor is None:
            self._handle_end_of_level()
        else:
            self._update_header()
        self.set_status("Undid the last redo -- restored the previous data.", "success")

    def _redo_last(self) -> None:
        if not self.last_capture_id:
            self.set_status("Nothing to redo yet.", "warning")
            return
        target = next((c for c in self.captures_cache if c["capture_id"] == self.last_capture_id), None)
        if target is None:
            self.set_status("Nothing to redo yet.", "warning")
            return
        self._record_undo_snapshot(target["level_index"], target["quantity"], [self.last_capture_id])
        cm.mark_superseded(self.captures_csv, self.last_capture_id)
        self.captures_cache = cm.read_captures(self.captures_csv)
        self._rebuild_run_grid()
        self.cursor = cm.next_cursor(
            self.captures_cache, self.grid, self.config["reference_node"],
            current_level_index=self.current_level_index,
        )
        self._update_header()
        self.set_status("Last capture marked for redo. Use Undo Last Redo to restore it.", "info")

    def _skip_node(self) -> None:
        if self.cursor is None or self.cursor["next_kind"] != "grid":
            self.set_status("Reference captures can't be skipped.", "warning")
            return
        record = cm.make_capture_record(
            campaign_id=self.config["campaign_id"], level_index=self.cursor["level_index"],
            variac_vac=self.current_level_variac, quantity=self.cursor["pass"], capture_kind="grid",
            node_label_=self.cursor["node_label"], grid=self.grid, sequence_index=self.cursor["sequence_index"],
            samples=[], duration_s=0.0, sensor_port="", note="skipped",
        )
        cm.append_capture(self.captures_csv, record)
        self.last_capture_id = record["capture_id"]
        self._refresh_after_capture(record)

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_btn.setText("Resume Pass" if self.paused else "Pause Pass")
        self.capture_btn.setEnabled(not self.paused and not self.warmup_active and self.cursor is not None)

    def _end_pass_early(self) -> None:
        if self.cursor is None or self.cursor["next_kind"] != "grid":
            return
        resp = QMessageBox.question(
            self, "End pass early",
            "Remaining grid nodes in this pass will be marked skipped, then the pass closes out "
            "with the reference-end capture. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return

        order = cm.serpentine_labels(self.grid)
        for idx in range(self.cursor["sequence_index"], len(order)):
            record = cm.make_capture_record(
                campaign_id=self.config["campaign_id"], level_index=self.cursor["level_index"],
                variac_vac=self.current_level_variac, quantity=self.cursor["pass"], capture_kind="grid",
                node_label_=order[idx], grid=self.grid, sequence_index=idx,
                samples=[], duration_s=0.0, sensor_port="", note="pass_ended_early",
            )
            cm.append_capture(self.captures_csv, record)
        self.captures_cache = cm.read_captures(self.captures_csv)
        self._rebuild_run_grid()
        self.cursor = cm.next_cursor(
            self.captures_cache, self.grid, self.config["reference_node"],
            current_level_index=self.current_level_index,
        )
        self._update_header()
        self.set_status("Remaining nodes skipped -- capture the reference-end node to close the pass.", "warning")

    def _on_redo_node_clicked(self, label: str) -> None:
        """Right-click a captured node in the Run tab grid to redo it. The
        old capture is superseded (never deleted, per plan section 5) and the
        cursor jumps to that node for a fresh capture. next_cursor() always
        picks the earliest gap in serpentine order, so once the redo is
        captured, advancing the cursor naturally snaps back to whichever node
        was current before the interruption -- no separate "resume" bookkeeping
        needed here."""
        if self.cursor is None or self.capture_in_progress or self.warmup_active:
            return
        state = self.run_grid.cell_state.get(label)
        if state not in ("captured", "flagged", "skipped"):
            return  # only already-captured cells can be jumped to for a redo
        level_index, quantity = self.cursor["level_index"], self.cursor["pass"]
        target = next(
            (c for c in cm.active_captures(self.captures_cache)
             if c["level_index"] == level_index and c["quantity"] == quantity
             and c["node_label"] == label and c["capture_kind"] == "grid"),
            None,
        )
        if target is None:
            return
        self._record_undo_snapshot(level_index, quantity, [target["capture_id"]])
        cm.mark_superseded(self.captures_csv, target["capture_id"])
        self.captures_cache = cm.read_captures(self.captures_csv)
        self.cursor = {
            "level_index": level_index, "pass": quantity, "next_kind": "grid",
            "sequence_index": target["sequence_index"], "node_label": label,
        }
        self._rebuild_run_grid()
        self._update_header()
        self.set_status(f"Redoing {label} -- capture will continue from where you left off afterward.", "info")

    # ------------------------------------------------------------------
    # Analysis tab
    # ------------------------------------------------------------------

    def _build_analysis_tab(self) -> QWidget:
        outer = QWidget()
        layout = QVBoxLayout(outer)

        ctrl_row = QHBoxLayout()
        self.analysis_level_combo = QComboBox()
        self.analysis_quantity_combo = QComboBox()
        self.analysis_quantity_combo.addItems(["lux", "irr"])
        self.analysis_mode_combo = QComboBox()
        self.analysis_mode_combo.addItems(["Value heatmap", "% deviation from panel-zone mean", "Lux/Irr ratio"])
        self.analysis_normalized_check = QCheckBox("Normalised (drift-corrected)")
        self.analysis_normalized_check.setChecked(True)
        for w in (self.analysis_level_combo, self.analysis_quantity_combo, self.analysis_mode_combo):
            w.currentIndexChanged.connect(self._guard(self._render_analysis))
        self.analysis_normalized_check.stateChanged.connect(self._guard(self._render_analysis))
        ctrl_row.addWidget(QLabel("Level"))
        ctrl_row.addWidget(self.analysis_level_combo)
        ctrl_row.addWidget(QLabel("Quantity"))
        ctrl_row.addWidget(self.analysis_quantity_combo)
        ctrl_row.addWidget(QLabel("View"))
        ctrl_row.addWidget(self.analysis_mode_combo)
        ctrl_row.addWidget(self.analysis_normalized_check)
        ctrl_row.addStretch()
        export_btn = QPushButton("Export")
        export_btn.setProperty("primary", True)
        export_btn.clicked.connect(self._guard(self._on_export))
        ctrl_row.addWidget(export_btn)
        layout.addLayout(ctrl_row)

        split = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(split, 1)

        fig = Figure(dpi=110, facecolor="#f0f2f5")
        self.analysis_canvas = FigureCanvas(fig)
        self.analysis_ax = fig.add_subplot(111)
        split.addWidget(self.analysis_canvas)

        bottom = QWidget()
        blay = QHBoxLayout(bottom)

        self.correction_table = QTableWidget()
        headers = ["Level", "VAC", "Qty", "Drift%", "Flag", "Panel Mean", "Sensor", "K", "Uniformity%", "CV%", "N"]
        self.correction_table.setColumnCount(len(headers))
        self.correction_table.setHorizontalHeaderLabels(headers)
        blay.addWidget(self.correction_table, 2)

        self.independence_lbl = QLabel("No analysis yet.")
        self.independence_lbl.setWordWrap(True)
        self.independence_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        blay.addWidget(self.independence_lbl, 1)

        split.addWidget(bottom)
        split.setSizes([500, 300])
        return outer

    def _populate_analysis_tab(self) -> None:
        if self.analysis is None:
            return
        self.analysis_level_combo.blockSignals(True)
        self.analysis_level_combo.clear()
        for level_index in sorted(self.analysis["levels"]):
            self.analysis_level_combo.addItem(f"Level {level_index}", level_index)
        self.analysis_level_combo.blockSignals(False)

        rows = self.analysis["correction_rows"]
        self.correction_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [
                row["level_index"], row["variac_vac"], row["quantity"],
                None if row["drift_pct"] is None else round(row["drift_pct"], 2),
                "YES" if row["drift_flagged"] else "",
                None if row["panel_mean"] is None else round(row["panel_mean"], 3),
                None if row["sensor_value"] is None else round(row["sensor_value"], 3),
                None if row["K"] is None else round(row["K"], 4),
                None if row["uniformity_pct"] is None else round(row["uniformity_pct"], 2),
                None if row["cv_pct"] is None else round(row["cv_pct"], 2),
                row["n_covered_nodes"],
            ]
            for c, v in enumerate(values):
                self.correction_table.setItem(r, c, QTableWidgetItem("" if v is None else str(v)))

        lines = []
        for quantity in ("lux", "irr"):
            report = self.analysis["independence"].get(quantity)
            if not report:
                continue
            lines.append(f"[{quantity.upper()}] {report['verdict']}")
            if report["linear_fit_vac"]:
                slope, intercept = report["linear_fit_vac"]
                lines.append(f"  K ~= {slope:.5f} * VAC + {intercept:.4f}")
        self.independence_lbl.setText("\n\n".join(lines) or "Not enough completed levels yet.")

        self._render_analysis()

    def _render_analysis(self) -> None:
        if self.analysis is None or self.analysis_level_combo.count() == 0:
            return
        level_index = self.analysis_level_combo.currentData()
        quantity = self.analysis_quantity_combo.currentText()
        mode = self.analysis_mode_combo.currentText()
        normalized = self.analysis_normalized_check.isChecked()

        self.analysis_ax.clear()
        level_data = self.analysis["levels"].get(level_index, {})

        if mode == "Lux/Irr ratio":
            values_by_node = self.analysis["ratio_maps"].get(level_index, {})
            title = f"Level {level_index}: lux / irr ratio"
        else:
            q_data = level_data.get(quantity)
            if not q_data:
                self.analysis_ax.set_title("No completed pass for this level/quantity")
                self.analysis_canvas.draw_idle()
                return
            values_by_node = q_data["norm_by_node"] if normalized else q_data["raw_by_node"]
            if mode == "% deviation from panel-zone mean":
                pm = q_data.get("panel_mean")
                values_by_node = (
                    {n: (v - pm) / pm * 100.0 for n, v in values_by_node.items()} if pm else values_by_node
                )
                title = f"Level {level_index} {quantity}: % deviation from panel-zone mean"
            else:
                title = f"Level {level_index} {quantity}: {'normalised' if normalized else 'raw'}"

        arr = np.full((self.grid.rows, self.grid.cols), np.nan)
        for label, value in values_by_node.items():
            col, row = cm.parse_node_label(label)
            arr[row, col] = value

        im = self.analysis_ax.imshow(arr, cmap="viridis", aspect="auto")
        self.analysis_ax.set_xticks(range(self.grid.cols))
        self.analysis_ax.set_xticklabels([cm.col_letter(i) for i in range(self.grid.cols)])
        self.analysis_ax.set_yticks(range(self.grid.rows))
        self.analysis_ax.set_yticklabels([str(i + 1) for i in range(self.grid.rows)])
        self.analysis_ax.set_title(title, fontsize=10)

        covered = self.analysis["covered_nodes"]
        if covered:
            cols = [cm.parse_node_label(n)[0] for n in covered]
            rows = [cm.parse_node_label(n)[1] for n in covered]
            self.analysis_ax.add_patch(
                Rectangle(
                    (min(cols) - 0.5, min(rows) - 0.5), max(cols) - min(cols) + 1, max(rows) - min(rows) + 1,
                    fill=False, edgecolor="white", linewidth=2,
                )
            )
        sensor_node = self.analysis["sensor_node"]
        if sensor_node:
            sc, sr = cm.parse_node_label(sensor_node)
            self.analysis_ax.plot(sc, sr, marker="*", color="red", markersize=14)

        self.analysis_canvas.figure.colorbar(im, ax=self.analysis_ax, shrink=0.8)
        self.analysis_canvas.figure.tight_layout()
        self.analysis_canvas.draw_idle()

    def _on_export(self) -> None:
        if self.analysis is None or self.campaign_dir is None:
            QMessageBox.warning(self, "Nothing to export", "Finish the campaign (or resume one) to run analysis first.")
            return
        adir = cm.analysis_dir(self.campaign_dir)
        adir.mkdir(parents=True, exist_ok=True)
        cm.write_correction_factors_csv(adir / "correction_factors.csv", self.analysis["correction_rows"])

        for level_index, quantities in self.analysis["levels"].items():
            for quantity, q_data in quantities.items():
                fig = Figure(dpi=150, facecolor="white")
                ax = fig.add_subplot(111)
                arr = np.full((self.grid.rows, self.grid.cols), np.nan)
                for label, value in q_data["norm_by_node"].items():
                    col, row = cm.parse_node_label(label)
                    arr[row, col] = value
                im = ax.imshow(arr, cmap="viridis", aspect="auto")
                ax.set_xticks(range(self.grid.cols))
                ax.set_xticklabels([cm.col_letter(i) for i in range(self.grid.cols)])
                ax.set_yticks(range(self.grid.rows))
                ax.set_yticklabels([str(i + 1) for i in range(self.grid.rows)])
                ax.set_title(f"Level {level_index} {quantity} (normalised)")
                fig.colorbar(im, ax=ax, shrink=0.8)
                fig.savefig(adir / f"heatmap_L{level_index}_{quantity}.png")

        open_in_explorer(str(adir))
        self.set_status(f"Analysis exported to {adir}", "success")


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightbox spatial calibration tool")
    parser.add_argument("--simulate", action="store_true", help="Use a synthetic light field instead of real sensors")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = LightboxCalibrationApp(simulate=args.simulate)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
