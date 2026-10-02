#!/usr/bin/env python3
"""Time-average tool -- standalone PyQt6 GUI.

Connects to the lux meter (LT68) and irradiance sensor, and on Measure
samples both as fast as they allow for an operator-set period, then shows
the average of the samples. Manually entered zero offsets are subtracted
from the live readings, the live graph and the averages.

The irradiance sensor streams on its own, so its rate is set by its
firmware; the lux meter is polled back-to-back (no pause between reads).

Run with --simulate to exercise the GUI with synthetic readings and no
hardware attached.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time

import serial.tools.list_ports

from PyQt6.QtCore import QLocale, QTimer
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import sensors
from ui_style import _STYLESHEET, _STAGE_STYLE, apply_light_theme
from zero_offset import ZeroOffsetCapture

MAX_DURATION_S = 3600.0
HISTORY_S = MAX_DURATION_S + 120.0  # keep a whole measurement on the graph
GRAPH_MIN_WINDOW_S = 60.0
MEASURE_POLL_MS = 50
SIM_TICK_MS = 20

_QUANTITIES = {
    "lux": {"name": "Lux", "unit": "lx", "fmt": "{:.2f}", "color": "#1a56db"},
    "irr": {"name": "Irradiance", "unit": "W/m²", "fmt": "{:.4f}", "color": "#c53030"},
}

_OFFSET_OK_STYLE = ""
_OFFSET_BAD_STYLE = "border: 2px solid #c53030;"


def _log(level: str, event: str, **fields) -> None:
    sensors._log(level, event, **fields)


def parse_offset(text: str) -> float | None:
    """Offset field text -> float, or None if empty/invalid (treated as 0)."""
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


class TimeAverageApp(QMainWindow):
    def __init__(self, simulate: bool = False):
        super().__init__()
        self.simulate = simulate

        self.lux_live = sensors.LiveValue()
        self.lux_history = sensors.TimeSeriesBuffer(max_age_s=HISTORY_S)
        self.irr_store = sensors.HistoryIrradiance(window_seconds=sensors.IRR_WINDOW_S, history_seconds=HISTORY_S)
        self._lux_stop = threading.Event()
        self._lux_thread = None
        self._irr_thread = None

        # Measurement state
        self.captures = {
            "lux": ZeroOffsetCapture(self.lux_history),
            "irr": ZeroOffsetCapture(self.irr_store.history),
        }
        self.duration_s = 0.0
        self.measure_offsets = {"lux": 0.0, "irr": 0.0}
        self.result_means: dict[str, float | None] = {"lux": None, "irr": None}

        self._build_ui()

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(150)
        self._live_timer.timeout.connect(self._guard(self._update_live_readouts))
        self._live_timer.start()

        self._ts_redraw_timer = QTimer(self)
        self._ts_redraw_timer.setInterval(sensors.TIMESERIES_REDRAW_MS)
        self._ts_redraw_timer.timeout.connect(self._guard(self._redraw_graph))
        self._ts_redraw_timer.start()

        self._measure_timer = QTimer(self)
        self._measure_timer.setInterval(MEASURE_POLL_MS)
        self._measure_timer.timeout.connect(self._guard(self._measure_tick))

        if self.simulate:
            self._sim_t0 = time.time()
            self._sim_timer = QTimer(self)
            self._sim_timer.setInterval(SIM_TICK_MS)
            self._sim_timer.timeout.connect(self._guard(self._sim_tick))
            self._sim_timer.start()
            self.set_status("Simulation mode -- synthetic readings, no hardware needed.", "info")
        else:
            self._auto_connect_sensors()

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
        self._lux_stop.set()
        event.accept()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setWindowTitle("Time Average" + (" [SIMULATE]" if self.simulate else ""))
        self.setStyleSheet(_STYLESHEET)
        self.resize(1200, 860)

        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        self.status_badge = QLabel("")
        self.status_badge.setWordWrap(True)
        layout.addWidget(self.status_badge)

        # Sensors
        if not self.simulate:
            port_box = QGroupBox("Sensors")
            prow = QHBoxLayout(port_box)
            pform = QFormLayout()
            self.lux_port_combo = QComboBox()
            self.irr_port_combo = QComboBox()
            pform.addRow("Lux meter (LT68)", self.lux_port_combo)
            pform.addRow("Irradiance sensor", self.irr_port_combo)
            prow.addLayout(pform, 1)
            refresh_btn = QPushButton("Refresh Ports")
            refresh_btn.clicked.connect(self._guard(self._refresh_port_lists))
            connect_btn = QPushButton("Connect")
            connect_btn.setProperty("primary", True)
            connect_btn.clicked.connect(self._guard(self._manual_connect_sensors))
            prow.addWidget(refresh_btn)
            prow.addWidget(connect_btn)
            layout.addWidget(port_box)

        # Inputs
        input_box = QGroupBox("Measurement")
        irow = QHBoxLayout(input_box)
        iform = QFormLayout()
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, MAX_DURATION_S)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setValue(10.0)
        self.duration_spin.setSuffix(" s")
        iform.addRow("Time period", self.duration_spin)

        validator = QDoubleValidator(self)
        validator.setLocale(QLocale.c())  # always "." as the decimal point
        self.offset_edits: dict[str, QLineEdit] = {}
        for key, q in _QUANTITIES.items():
            edit = QLineEdit("0")
            edit.setValidator(validator)
            edit.textChanged.connect(self._guard(self._validate_offsets))
            self.offset_edits[key] = edit
            iform.addRow(f"{q['name']} zero offset ({q['unit']})", edit)
        irow.addLayout(iform, 1)

        self.measure_btn = QPushButton("Measure")
        self.measure_btn.setProperty("primary", True)
        self.measure_btn.clicked.connect(self._guard(self._start_measure))
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setProperty("danger", True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._guard(self._stop_early))
        irow.addWidget(self.measure_btn)
        irow.addWidget(self.stop_btn)
        layout.addWidget(input_box)

        # Live readouts (offset-corrected, raw shown small)
        live_row = QHBoxLayout()
        self.live_lbls: dict[str, QLabel] = {}
        self.raw_lbls: dict[str, QLabel] = {}
        self.badges: dict[str, QLabel] = {}
        for key, q in _QUANTITIES.items():
            col = QVBoxLayout()
            badge = QLabel(f"● {q['name']}: --")
            val = QLabel(f"-- {q['unit']}")
            val.setStyleSheet(f"font-size:22pt; font-weight:bold; color:{q['color']};")
            raw = QLabel("raw --")
            raw.setStyleSheet("color:#555555;")
            col.addWidget(badge)
            col.addWidget(val)
            col.addWidget(raw)
            live_row.addLayout(col)
            live_row.addSpacing(40)
            self.live_lbls[key], self.raw_lbls[key], self.badges[key] = val, raw, badge
        live_row.addStretch()
        layout.addLayout(live_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        layout.addWidget(self.progress)

        # Results
        res_box = QGroupBox("Average")
        grid = QGridLayout(res_box)
        headers = ["", "Average (offset-corrected)", "Raw average", "Offset", "Std dev", "Samples", "Rate"]
        for c, h in enumerate(headers):
            lbl = QLabel(f"<b>{h}</b>")
            grid.addWidget(lbl, 0, c)
        self.result_cells: dict[str, list[QLabel]] = {}
        for r, (key, q) in enumerate(_QUANTITIES.items(), start=1):
            grid.addWidget(QLabel(f"<b>{q['name']}</b>"), r, 0)
            cells = []
            for c in range(1, len(headers)):
                lbl = QLabel("--")
                if c == 1:
                    lbl.setStyleSheet(f"font-size:16pt; font-weight:bold; color:{q['color']};")
                grid.addWidget(lbl, r, c)
                cells.append(lbl)
            self.result_cells[key] = cells
        self.result_note = QLabel("")
        grid.addWidget(self.result_note, len(_QUANTITIES) + 1, 0, 1, len(headers))
        layout.addWidget(res_box)

        # Live graph
        fig = Figure(dpi=100, facecolor="#f0f2f5")
        self.canvas = FigureCanvas(fig)
        self.axes = {"lux": fig.add_subplot(1, 2, 1), "irr": fig.add_subplot(1, 2, 2)}
        self._style_axes()
        fig.tight_layout()
        layout.addWidget(self.canvas, 1)

        if not self.simulate:
            self._refresh_port_lists()

    def _style_axes(self) -> None:
        for key, ax in self.axes.items():
            q = _QUANTITIES[key]
            ax.set_title(f"{q['name']} (live, offset-corrected)", fontsize=9)
            ax.set_xlabel("Seconds ago", fontsize=8)
            ax.set_ylabel(q["unit"], fontsize=8)
            ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Offsets
    # ------------------------------------------------------------------

    def _validate_offsets(self) -> None:
        for edit in self.offset_edits.values():
            ok = parse_offset(edit.text()) is not None
            edit.setStyleSheet(_OFFSET_OK_STYLE if ok else _OFFSET_BAD_STYLE)

    def current_offset(self, key: str) -> float:
        value = parse_offset(self.offset_edits[key].text())
        return 0.0 if value is None else value

    # ------------------------------------------------------------------
    # Sensor connection (same pattern as lightbox_calibration.py)
    # ------------------------------------------------------------------

    def _refresh_port_lists(self) -> dict:
        ports = list(serial.tools.list_ports.comports())
        detected = sensors.detect_sensor_ports()
        for combo, name in ((self.lux_port_combo, "lux"), (self.irr_port_combo, "irr")):
            cached = sensors.load_cached_port(name)
            combo.blockSignals(True)
            combo.clear()
            seen = set()
            for p in ports:
                label = f"{p.device} — {p.description}" if p.description else p.device
                combo.addItem(label, p.device)
                seen.add(p.device)
            if cached and cached not in seen:
                combo.addItem(f"{cached} (not currently connected)", cached)
            preferred = detected[name] or cached
            if preferred:
                idx = combo.findData(preferred)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        return detected

    def _auto_connect_sensors(self) -> None:
        detected = self._refresh_port_lists()
        if detected["lux"] or detected["irr"]:
            self.set_status("Auto-detected sensor(s) -- connecting…", "info")
            self._connect_sensors()
        else:
            self.set_status(
                "Could not auto-detect the lux meter (CP210x) or irradiance sensor (CH9102) -- "
                "check they're plugged in, or pick their ports above and press Connect.",
                "warning",
            )

    def _manual_connect_sensors(self) -> None:
        self.set_status("Connecting sensor(s)…", "info")
        self._connect_sensors()

    def _connect_sensors(self) -> None:
        lux_port = self.lux_port_combo.currentData() or self.lux_port_combo.currentText().strip()
        irr_port = self.irr_port_combo.currentData() or self.irr_port_combo.currentText().strip()
        if not lux_port and not irr_port:
            self.set_status("Pick at least one sensor COM port before connecting.", "warning")
            return

        if lux_port and not (self._lux_thread is not None and self._lux_thread.is_alive()):
            if sensors.port_in_use(lux_port):
                self.set_status(f"{lux_port} is in use -- close the other sensor apps first.", "error")
            else:
                sensors.save_cached_port("lux", lux_port)
                self._lux_stop.clear()
                self._lux_thread = threading.Thread(
                    target=sensors.lux_poll_loop,
                    args=(lux_port, self.lux_live, self._lux_stop, self.lux_history),
                    kwargs={"interval_s": 0.0},  # read back-to-back: as fast as the meter allows
                    daemon=True,
                )
                self._lux_thread.start()

        if irr_port and not (self._irr_thread is not None and self._irr_thread.is_alive()):
            if sensors.port_in_use(irr_port):
                self.set_status(f"{irr_port} is in use -- close the other sensor apps first.", "error")
            else:
                sensors.save_cached_port("irr", irr_port)
                from listen import serial_reader
                self._irr_thread = threading.Thread(
                    target=serial_reader,
                    args=(irr_port, sensors.IRR_BAUD_DEFAULT, self.irr_store, 10**9, 1.0, 2.0, 30.0, _log),
                    daemon=True,
                )
                self._irr_thread.start()

    # ------------------------------------------------------------------
    # Live readouts + graph
    # ------------------------------------------------------------------

    def _latest(self, key: str):
        if key == "lux":
            value, ts, err = self.lux_live.snapshot()
            return value, ts, err
        value, ts = self.irr_store.latest()
        return value, ts, None

    def _update_live_readouts(self) -> None:
        now = time.time()
        for key, q in _QUANTITIES.items():
            value, ts, err = self._latest(key)
            badge = self.badges[key]
            if value is not None and (ts is None or now - ts <= sensors.LIVE_STALE_S):
                corrected = value - self.current_offset(key)
                self.live_lbls[key].setText(f"{q['fmt'].format(corrected)} {q['unit']}")
                self.raw_lbls[key].setText(f"raw {q['fmt'].format(value)} {q['unit']}")
                badge.setText(f"● {q['name']}: live")
                badge.setStyleSheet("color:#1b5e20; font-weight:bold;")
            else:
                self.live_lbls[key].setText(f"-- {q['unit']}")
                self.raw_lbls[key].setText("raw --")
                badge.setText(f"● {q['name']}: {'error' if err else '--'}")
                badge.setStyleSheet(f"color:{'#9b1c1c' if err else '#8a5300'}; font-weight:bold;")

    def _redraw_graph(self) -> None:
        now = time.time()
        window_s = max(GRAPH_MIN_WINDOW_S, self.duration_spin.value() * 1.5)
        histories = {"lux": self.lux_history, "irr": self.irr_store.history}
        for key, ax in self.axes.items():
            ax.clear()
            times, values = histories[key].snapshot()
            offset = self.current_offset(key)
            xs, ys = [], []
            for t, v in zip(times, values):
                if now - t <= window_s and v is not None:
                    xs.append(t - now)
                    ys.append(v - offset)
            if xs:
                ax.plot(xs, ys, color=_QUANTITIES[key]["color"], linewidth=1.0)
            cap = self.captures[key]
            if cap.started_at is not None:
                end = cap.stopped_at if cap.stopped_at is not None else now
                ax.axvspan(cap.started_at - now, end - now, color="#f2a900", alpha=0.15)
                if not cap.running and self.result_means[key] is not None:
                    ax.axhline(self.result_means[key], color="#1a1a1a", linestyle="--", linewidth=1.0)
            ax.set_xlim(-window_s, 0)
        self._style_axes()
        self.canvas.figure.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def _measuring(self) -> bool:
        return any(c.running for c in self.captures.values())

    def _start_measure(self) -> None:
        self.duration_s = self.duration_spin.value()
        self.measure_offsets = {key: self.current_offset(key) for key in _QUANTITIES}
        self.result_means = {key: None for key in _QUANTITIES}
        now = time.time()
        for cap in self.captures.values():
            cap.start(now=now)
        for cells in self.result_cells.values():
            for lbl in cells:
                lbl.setText("--")
        self.result_note.setText("")
        self.measure_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.duration_spin.setEnabled(False)
        self.set_status(f"Measuring for {self.duration_s:g} s…", "info")
        self._measure_timer.start()

    def _measure_tick(self) -> None:
        for cap in self.captures.values():
            cap.poll()
        elapsed = self.captures["lux"].elapsed()
        frac = min(1.0, elapsed / self.duration_s) if self.duration_s > 0 else 1.0
        self.progress.setValue(int(frac * 1000))
        counts = ", ".join(f"{_QUANTITIES[k]['name']} {c.n}" for k, c in self.captures.items())
        self.progress.setFormat(f"{elapsed:.1f} / {self.duration_s:g} s  ({counts} samples)")
        if elapsed >= self.duration_s:
            self._finish_measure(stopped_early=False)

    def _stop_early(self) -> None:
        if self._measuring():
            self._finish_measure(stopped_early=True)

    def _finish_measure(self, stopped_early: bool) -> None:
        self._measure_timer.stop()
        now = time.time()
        if not stopped_early:
            # End exactly at the requested period, not at the next timer tick.
            now = self.captures["lux"].started_at + self.duration_s
        for cap in self.captures.values():
            cap.stop(now=now)
            # stop() polls first; drop anything that landed after the end time.
            while cap.times and cap.times[-1] > cap.stopped_at:
                cap.times.pop()
                cap.values.pop()

        missing = []
        for key, q in _QUANTITIES.items():
            cap = self.captures[key]
            cells = self.result_cells[key]
            offset = self.measure_offsets[key]
            fmt, unit = q["fmt"], q["unit"]
            elapsed = cap.elapsed()
            cells[2].setText(f"{fmt.format(offset)} {unit}")
            if cap.n == 0:
                cells[0].setText("no samples")
                missing.append(q["name"])
                continue
            raw_mean = cap.mean()
            self.result_means[key] = raw_mean - offset
            sd = cap.stdev()
            cells[0].setText(f"{fmt.format(raw_mean - offset)} {unit}")
            cells[1].setText(f"{fmt.format(raw_mean)} {unit}")
            cells[3].setText(f"{fmt.format(sd)} {unit}" if sd is not None else "--")
            cells[4].setText(str(cap.n))
            cells[5].setText(f"{cap.n / elapsed:.1f} Hz" if elapsed > 0 else "--")

        elapsed = self.captures["lux"].elapsed()
        stamp = time.strftime("%H:%M:%S", time.localtime(self.captures["lux"].started_at))
        note = f"Started {stamp}, {elapsed:.1f} s"
        if stopped_early:
            note += f" (stopped early, of {self.duration_s:g} s requested)"
        self.result_note.setText(note)

        self.progress.setValue(1000 if not stopped_early else self.progress.value())
        self.progress.setFormat("Done" + (" (stopped early)" if stopped_early else ""))
        self.measure_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.duration_spin.setEnabled(True)
        if missing:
            self.set_status(f"Measurement done -- no samples from: {', '.join(missing)}.", "warning")
        else:
            self.set_status("Measurement done.", "success")

    # ------------------------------------------------------------------
    # Simulation driver
    # ------------------------------------------------------------------

    def _sim_tick(self) -> None:
        t = time.time() - self._sim_t0
        drift = 1.0 + 0.02 * math.sin(t / 15.0)
        lux_val = 250.0 * drift * (1.0 + random.gauss(0.0, 0.01))
        irr_val = 1.8 * drift * (1.0 + random.gauss(0.0, 0.01))
        self.lux_live.set(lux_val)
        self.lux_history.add(lux_val)
        self.irr_store.add(irr_val)


def main() -> int:
    parser = argparse.ArgumentParser(description="Time-average the lux and irradiance sensors")
    parser.add_argument("--simulate", action="store_true", help="Use synthetic readings instead of real sensors")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    apply_light_theme(app)
    window = TimeAverageApp(simulate=args.simulate)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
