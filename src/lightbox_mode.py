"""Lightbox mode for the Low Light IV app: pure logic plus its dialogs.

A round of measurements targets one setpoint: the average irradiance over the
panel footprint centred on F4 (width along A-K, height along rows), taken
from a calibration campaign folder the operator chooses. For each new panel
size the app gives a starting VAC and the reading the irradiance sensor should
show at its node; the operator trims the variac until the live reading
matches (the variac readout alone is only good to about +/-10 % at low
light). Panels of the same size reuse the light setting. At save time the
irradiance and lux readings at their nodes are compared with what the
calibration predicts.

The maths is calibration_report.py's (vac_for_target / expected_at_vac), so
the app and the report always agree.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

import calibration_model as cm
import calibration_report as cr

DARK_SETTLE_S = 2.0
DARK_COLLECT_S = 5.0
DARK_OFFSET_WARN_WM2 = 0.05
TRIM_LIVE_WINDOW_S = 2.0
DEFAULT_IRR_NODE = "B6"
DEFAULT_LUX_NODE = "J6"


# ---------------------------------------------------------------------------
# Pure logic (no Qt)
# ---------------------------------------------------------------------------


class Calibration:
    """A loaded calibration campaign folder."""

    def __init__(self, folder):
        self.folder = Path(folder)
        if not (self.folder / "campaign.json").exists() or not (self.folder / "captures.csv").exists():
            raise ValueError(f"{self.folder} is not a calibration campaign folder (needs campaign.json and captures.csv).")
        self.config, self.grid, self.passes = cr.load_calibration(self.folder)
        self.irr_offset = cr.campaign_irr_offset(self.config)
        self.campaign_id = self.config.get("campaign_id") or self.folder.name
        self.vacs = sorted({p["vac"] for p in self.passes.values()})
        self.exclusions = cr.load_exclusions(self.folder, self.grid)

    @property
    def summary(self) -> str:
        ex = f", {sum(len(e['nodes']) for e in self.exclusions)} excluded node(s)" if self.exclusions else ""
        return (f"{self.campaign_id} ({len(self.vacs)} levels, {self.vacs[0]:g}-{self.vacs[-1]:g} VAC{ex}, "
                f"irr zero offset {self.irr_offset:.4g} W/m²)")

    def node_labels(self) -> list[str]:
        return cm.all_node_labels(self.grid)

    def light_setting(self, setpoint: float, w_mm: float, h_mm: float, irr_node: str) -> dict:
        """Starting VAC for a panel-average `setpoint` over w x h, and what the
        irradiance sensor at irr_node should read there (offset-subtracted).
        Raises ValueError (out of range, bad node, ...)."""
        res = cr.vac_for_target(self.passes, self.grid, w_mm, h_mm, setpoint, "irr", {"irr": irr_node})
        return {
            "vac_target": res["vac"],
            "vac_unc_v": res["vac_unc_v"],
            "bracket": res["bracket"],
            "irr_node": irr_node,
            "irr_target": res["sensors"]["irr"]["value"],
            "irr_target_drift_pct": res["sensors"]["irr"]["drift_pct"],
            "lux_panel_avg": res["lux_avg"],
        }

    def expected(self, vac: float, irr_node: str, lux_node: str, w_mm: float, h_mm: float) -> dict:
        return cr.expected_at_vac(self.passes, self.grid, vac, {"irr": irr_node, "lux": lux_node}, w_mm, h_mm)

    def vac_for_setpoint(self, setpoint: float, w_mm: float, h_mm: float) -> tuple[float, bool]:
        """(VAC, extrapolated?) for a panel-average irradiance setpoint, exactly
        as light_setting() works it out. Outside the calibrated range the power
        law of the two end levels is carried on instead of refusing."""
        pts = sorted((p["vac"], cr.panel_average(p["filled"], self.grid, w_mm, h_mm))
                     for (_level, q), p in self.passes.items() if q == "irr")
        if pts[0][1] <= setpoint <= pts[-1][1]:
            return cr.vac_for_target(self.passes, self.grid, w_mm, h_mm, setpoint, "irr")["vac"], False
        (v0, y0), (v1, y1) = pts[:2] if setpoint < pts[0][1] else pts[-2:]
        return cr._loglog(v0, y0, v1, y1, setpoint), True

    def expected_at_node(self, vac: float, quantity: str, node: str) -> tuple[float, bool]:
        """(expected reading, extrapolated?) for one sensor at `node` and
        `vac`. Used when a saved row's node is corrected after the fact, so a
        VAC just outside the calibrated range (rows set below the lowest
        level) is extrapolated rather than refused."""
        extrapolated = not (self.vacs[0] - 1e-9 <= vac <= self.vacs[-1] + 1e-9)
        exp = cr.expected_at_vac(self.passes, self.grid, vac, {quantity: node}, extrapolate=True)
        return exp["nodes"][quantity]["value"], extrapolated


@dataclass
class Round:
    """State of one setpoint round. `size`/`light` are None until the first
    panel of the round has had its light set."""
    setpoint: float
    dark_offset: float
    dark_offset_source: str = "measured"
    size: tuple | None = None
    light: dict | None = None      # Calibration.light_setting() + "vac_set"
    last_panel: str = ""
    started_at: float = field(default_factory=time.time)

    def needs_light_setting(self, w_mm: float, h_mm: float) -> bool:
        return self.light is None or self.size != (float(w_mm), float(h_mm))


def deviation_pct(measured, expected):
    if measured is None or expected in (None, 0):
        return None
    return (measured - expected) / expected * 100.0


def window_mean(history, t0: float, t1: float):
    """(mean, n) of a TimeSeriesBuffer's samples with t0 <= t <= t1."""
    times, values = history.snapshot()
    vals = [v for t, v in zip(times, values) if t0 <= t <= t1]
    return (float(np.mean(vals)) if vals else None), len(vals)


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------


class NextPanelDialog(QDialog):
    """Panel name plus the two sensor nodes, pre-filled from the last panel."""

    def __init__(self, parent, panel_name: str, irr_node: str, lux_node: str, node_labels: list[str]):
        super().__init__(parent)
        self.setWindowTitle("Next Panel")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(panel_name)
        self.irr_combo = QComboBox()
        self.lux_combo = QComboBox()
        for combo, node in ((self.irr_combo, irr_node), (self.lux_combo, lux_node)):
            combo.addItems(node_labels)
            if node in node_labels:
                combo.setCurrentText(node)
        form.addRow("Panel Name:", self.name_edit)
        form.addRow("Irradiance sensor node:", self.irr_combo)
        form.addRow("Lux sensor node:", self.lux_combo)
        lay.addLayout(form)
        self.err = QLabel("")
        self.err.setStyleSheet("color:#c53030;")
        lay.addWidget(self.err)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.name_edit.selectAll()

    def _accept(self):
        if not self.name_edit.text().strip():
            self.err.setText("Panel Name is required.")
            return
        self.accept()

    def values(self):
        return self.name_edit.text().strip(), self.irr_combo.currentText(), self.lux_combo.currentText()


class DarkOffsetDialog(QDialog):
    """Cover the irradiance sensor, wait DARK_SETTLE_S, average DARK_COLLECT_S
    of samples. `self.offset` / `self.source` are set on accept.
    stored_offset: (value, captured_at) from the Irr Zero Offset tab, offered
    as another choice when given."""

    def __init__(self, parent, history, reference_offset: float, min_samples: int, stored_offset=None):
        super().__init__(parent)
        self.setWindowTitle("Irradiance Dark Offset")
        self.history = history
        self.reference_offset = reference_offset
        self.min_samples = min_samples
        self.offset = None
        self.source = ""
        self._t0 = None

        lay = QVBoxLayout(self)
        self.msg = QLabel(
            "Cover the irradiance sensor completely (no light at all), then click Start.\n"
            f"The app waits {DARK_SETTLE_S:g} s, then averages {DARK_COLLECT_S:g} s of readings. "
            "That reading is subtracted from every irradiance reading in this round."
        )
        self.msg.setWordWrap(True)
        lay.addWidget(self.msg)
        self.status = QLabel("")
        self.status.setStyleSheet("font-size:12pt; font-weight:bold;")
        lay.addWidget(self.status)

        row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setProperty("primary", True)
        self.start_btn.clicked.connect(self._start)
        self.use_btn = QPushButton("Use This Offset")
        self.use_btn.setEnabled(False)
        self.use_btn.clicked.connect(self._use_measured)
        self.fallback_btn = QPushButton(f"Use Calibration Offset ({reference_offset:g})")
        self.fallback_btn.setToolTip("Skip the dark reading and use the calibration campaign's fixed offset")
        self.fallback_btn.clicked.connect(self._use_reference)
        buttons = [self.start_btn, self.use_btn]
        self.stored_offset = stored_offset
        if stored_offset is not None:
            value, captured_at = stored_offset
            self.stored_btn = QPushButton(f"Use Zero-Offset Tab ({value:.4f}{', ' + captured_at if captured_at else ''})")
            self.stored_btn.setToolTip("Skip the dark reading and use the offset captured on the Irr Zero Offset tab")
            self.stored_btn.clicked.connect(self._use_stored)
            buttons.append(self.stored_btn)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        for b in buttons + [self.fallback_btn, cancel]:
            row.addWidget(b)
        lay.addLayout(row)

        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._tick)

    def _start(self):
        self._t0 = time.time()
        self._measured = None
        self.start_btn.setEnabled(False)
        self.use_btn.setEnabled(False)
        self._timer.start()
        self._tick()

    def _tick(self):
        elapsed = time.time() - self._t0
        if elapsed < DARK_SETTLE_S:
            self.status.setText(f"Settling… {DARK_SETTLE_S - elapsed:.1f} s")
            return
        t_start = self._t0 + DARK_SETTLE_S
        mean, n = window_mean(self.history, t_start, time.time())
        if elapsed < DARK_SETTLE_S + DARK_COLLECT_S:
            self.status.setText(f"Measuring… {DARK_SETTLE_S + DARK_COLLECT_S - elapsed:.1f} s, {n} sample(s)")
            return
        self._timer.stop()
        self.start_btn.setText("Retry")
        self.start_btn.setEnabled(True)
        if n < self.min_samples:
            self.status.setText(f"Only {n} sample(s) — is the irradiance sensor connected? Retry, or use the calibration offset.")
            self.status.setStyleSheet("font-size:11pt; font-weight:bold; color:#c53030;")
            return
        self._measured = mean
        text = f"Dark offset: {mean:.4f} W/m² ({n} samples)"
        if abs(mean - self.reference_offset) > DARK_OFFSET_WARN_WM2:
            text += (f"\nThis is {mean - self.reference_offset:+.3f} W/m² from the calibration's "
                     f"{self.reference_offset:g}. Is the sensor fully covered?")
            self.status.setStyleSheet("font-size:11pt; font-weight:bold; color:#8a5300;")
        else:
            self.status.setStyleSheet("font-size:12pt; font-weight:bold; color:#1b8a5a;")
        self.status.setText(text)
        self.use_btn.setEnabled(True)

    def _use_measured(self):
        self.offset, self.source = self._measured, "measured"
        self._timer.stop()
        self.accept()

    def _use_stored(self):
        self.offset, self.source = self.stored_offset[0], "zero-offset tab"
        self._timer.stop()
        self.accept()

    def _use_reference(self):
        self.offset, self.source = self.reference_offset, "calibration"
        self._timer.stop()
        self.accept()


class TrimDialog(QDialog):
    """Set the variac to the starting VAC, then trim until the irradiance
    sensor (live, offset-subtracted) matches the target. `self.vac_set` is
    set on accept."""

    def __init__(self, parent, panel_name: str, size: tuple, setpoint: float, light: dict,
                 dark_offset: float, tolerance_pct: float, live_fn):
        super().__init__(parent)
        self.setWindowTitle("Set the Light")
        self.setMinimumWidth(460)
        self.target = light["irr_target"]
        self.dark_offset = dark_offset
        self.tolerance_pct = tolerance_pct
        self.live_fn = live_fn
        self.vac_set = None
        self._last_dev = None

        lay = QVBoxLayout(self)
        w, h = size
        intro = QLabel(
            f"<b>{panel_name}</b>: {w:g} × {h:g} mm, setpoint {setpoint:g} W/m² (panel average).<br><br>"
            f"1. Set the variac to about <b>{light['vac_target']:.1f} VAC</b> (± {light['vac_unc_v']:.2f} V from calibration drift).<br>"
            f"2. Trim the variac until the irradiance sensor at <b>{light['irr_node']}</b> reads "
            f"<b>{self.target:.4g} W/m²</b> (sensor display ≈ {self.target + dark_offset:.4g} with the "
            f"{dark_offset:.3g} dark offset)."
        )
        intro.setWordWrap(True)
        lay.addWidget(intro)

        self.live_lbl = QLabel("Live: --")
        self.live_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_lbl.setStyleSheet("font-size:16pt; font-weight:bold; padding:8px;")
        lay.addWidget(self.live_lbl)

        form = QFormLayout()
        self.vac_edit = QLineEdit(f"{light['vac_target']:.1f}")
        self.vac_edit.setToolTip("The variac reading after trimming. Logged with every row; not used in the maths.")
        form.addRow("VAC as set (after trimming):", self.vac_edit)
        lay.addLayout(form)
        self.err = QLabel("")
        self.err.setStyleSheet("color:#c53030;")
        lay.addWidget(self.err)

        buttons = QDialogButtonBox()
        confirm = buttons.addButton("Confirm — Light Is Set", QDialogButtonBox.ButtonRole.AcceptRole)
        confirm.setProperty("primary", True)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._confirm)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._tick()

    def _tick(self):
        raw, n = self.live_fn()
        if raw is None:
            self._last_dev = None
            self.live_lbl.setText("Live: no irradiance readings in the last 2 s")
            self.live_lbl.setStyleSheet("font-size:13pt; font-weight:bold; padding:8px; color:#c53030;")
            return
        value = raw - self.dark_offset
        dev = deviation_pct(value, self.target)
        self._last_dev = dev
        ok = abs(dev) <= self.tolerance_pct
        color = "#1b8a5a" if ok else ("#c53030" if abs(dev) > 3 * self.tolerance_pct else "#8a5300")
        hint = "on target" if ok else ("turn the variac UP" if dev < 0 else "turn the variac DOWN")
        self.live_lbl.setText(f"Live: {value:.4g} W/m²   {dev:+.1f}%   ({hint})")
        self.live_lbl.setStyleSheet(f"font-size:16pt; font-weight:bold; padding:8px; color:{color};")

    def _confirm(self):
        try:
            vac = float(self.vac_edit.text().strip())
            if not vac > 0:
                raise ValueError
        except ValueError:
            self.err.setText("Enter the VAC the variac shows (a number).")
            return
        if self._last_dev is None or abs(self._last_dev) > self.tolerance_pct:
            off = "no live reading" if self._last_dev is None else f"{self._last_dev:+.1f}% off target"
            resp = QMessageBox.question(
                self, "Not on target",
                f"The irradiance sensor is {off} (tolerance ± {self.tolerance_pct:g}%). Use this light setting anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return
        self.vac_set = vac
        self._timer.stop()
        self.accept()

    def reject(self):
        self._timer.stop()
        super().reject()
