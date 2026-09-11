#!/usr/bin/env python3
"""Unified PyQt6 GUI for Low Light IV Testing — Voltaic Systems.

Single-window replacement for the old three-terminal workflow
(listen.py background daemon + tkinter popup / blocking plots in
IV_curve_CURRENT_V3.py + excel_capture.py). Reuses the existing
hardware/session logic by import; only two small, backward-compatible
signature additions were made to IV_curve_CURRENT_V3.py (see plan).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import csv
import functools
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports

from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
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
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import norm_analysis
from triplett_lt68_probe_v3 import LT68
from listen import RollingIrradiance, serial_reader, log_event
from IV_curve_CURRENT_V3 import (
    append_summary_csv,
    build_clipboard_row,
    connect_otii_session,
    copy_to_clipboard,
    derive_panel_type,
    fetch_recording_channels,
    harvest,
    make_row,
    moving_average,
    read_max_run_index,
    render_iv_curve,
    restart_otii_app,
    rewrite_summary_csv,
    short_circuit,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "report" / "LowLightTesting_summary.csv"

# Constants carried over unchanged from listen.py / IV_curve_CURRENT_V3.py defaults,
# except the poll/redraw cadences below -- tightened (2026-09) to cut the
# perceived lag between the physical sensors and what's on screen. The lux
# meter's own settle time (LUX_SETTLE_S) is the real floor on read rate and
# is left alone (it's an accuracy setting, not just latency); the polling
# loop's *extra* wait between reads is what actually got tightened.
LUX_TIMEOUT_S = 1.0
LUX_SETTLE_S = 0.25
LUX_POLL_INTERVAL_S = 0.3
IRR_BAUD_DEFAULT = 115200
IRR_WINDOW_S = 10.0
MIN_IRR_SAMPLES = 3
LIVE_STALE_S = 5.0
TIMESERIES_HISTORY_S = 600.0  # how much lux/irradiance history the timeseries plot keeps
TIMESERIES_REDRAW_MS = 400

# Stable, deterministic color assignment (by sorted group label) for the
# Analysis tab's 4 graphs, so a given panel type keeps the same color
# across redraws and across all 4 subplots.
_ANALYSIS_PALETTE = [
    "#1a56db", "#c53030", "#1b8a5a", "#8a5300", "#6b46c1",
    "#0f7a8c", "#b83280", "#4a5568", "#c05621", "#2f855a",
]

_COLUMNS = [
    "Panel Name", "Time", "Wp (W)", "Voc (V)", "Isc (A)", "Vp (V)", "Ip (A)",
    "Light Meter", "SI (GUI) (W/m²)", "Temp (°C)", "Lux (lx)", "Lux StdDev (lx)", "Lux 3σ/Avg (%)",
    "Irradiance (W/m²)", "Irr StdDev (W/m²)", "Irr 3σ/Avg (%)", "Notes",
]
_EDITABLE_COLS = {0, 7, 8, 9, 16}
_FIELD_MAP = {0: "panel_name", 7: "light_meter", 8: "irradiance_gui_input", 9: "panel_temp_c", 16: "notes"}


# ---------------------------------------------------------------------------
# Sensor auto-detection.
#
# The two sensors show up in Device Manager by their USB-serial bridge chip,
# not by a friendly name: the Triplett lux meter enumerates as a Silicon
# Labs "CP210x USB to UART Bridge", and the irradiance sensor enumerates as
# a WCH "USB-Enhanced-SERIAL CH9102". Matching on those substrings lets the
# GUI find the right COM port automatically instead of the operator having
# to look it up in Device Manager every time.
# ---------------------------------------------------------------------------

_LUX_DEVICE_PATTERN = re.compile(r"cp210", re.IGNORECASE)
_IRR_DEVICE_PATTERN = re.compile(r"ch910", re.IGNORECASE)


def detect_sensor_ports() -> dict:
    """Return {"lux": "COMx" or None, "irr": "COMy" or None} by matching
    each enumerated serial port's USB descriptor against the known bridge
    chips. If more than one port matches a pattern, the first one found
    wins (documented as a known limitation for future multi-device setups)."""
    detected = {"lux": None, "irr": None}
    for p in serial.tools.list_ports.comports():
        haystack = " ".join(filter(None, [p.description, p.manufacturer, getattr(p, "product", None)]))
        if detected["lux"] is None and _LUX_DEVICE_PATTERN.search(haystack):
            detected["lux"] = p.device
        if detected["irr"] is None and _IRR_DEVICE_PATTERN.search(haystack):
            detected["irr"] = p.device
    return detected


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


def _log(level: str, event: str, **fields) -> None:
    log_event(level, event, **fields)


# ---------------------------------------------------------------------------
# Async worker — a dedicated background thread running a plain, standard
# asyncio event loop, completely separate from Qt's event loop.
#
# 2026-09 field debugging, round 1: connect_otii_session() called directly
# (plain synchronous script, no Qt/asyncio at all) succeeds instantly. Called
# via asyncio.to_thread() inside this app's qasync event loop, it never
# completed -- the coroutine hung, alongside a one-time "QObject::startTimer:
# Timers can only be used with threads started with QThread" warning. That
# was patched by giving call_with_timeout() its own thread+Qt-signal bridge
# instead of relying on qasync's executor/future-chaining internals.
#
# Round 2: even with that fix, clicking "Connect to Otii" still did nothing
# at all -- the status badge never even changed to "Connecting...", meaning
# the task's *first line* never ran. Meanwhile this app's regular QTimers
# (the live sensor readout, the timeseries redraw) kept firing correctly the
# whole time, proving Qt's own native event loop was healthy. That isolates
# the problem precisely: qasync's own asyncio-loop task scheduling
# (ensure_future()/create_task() -> its Qt-timer-based _SimpleTimer.
# add_callback()/startTimer()) is what's unreliable in this app's actual
# runtime -- not just the executor-completion path fixed in round 1.
#
# The fix is to stop depending on qasync's asyncio-loop integration
# entirely. AsyncWorker owns a bare `asyncio.new_event_loop()` on its own
# thread -- standard library, well-tested, nothing Qt-specific about its
# task scheduling. The main thread keeps running Qt's own native event loop
# (QApplication.exec(), no qasync). Coroutines are submitted across threads
# via asyncio.run_coroutine_threadsafe() (the standard, documented mechanism
# for exactly this), and any code that needs to touch a Qt widget marshals
# back to the main thread via LowLightApp._on_main()/_call_on_main(), which
# uses a plain Qt signal -- the same well-established cross-thread pattern
# already used for harvest()'s progress updates in _ProgressBridge.
# ---------------------------------------------------------------------------


class OtiiCommsTimeout(RuntimeError):
    """Raised when an Otii call does not respond within its allotted timeout."""


class AsyncWorker:
    """Runs a plain asyncio event loop on a dedicated background thread."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro) -> concurrent.futures.Future:
        """Schedule a coroutine on the worker loop from any thread."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


async def call_with_timeout(fn, *args, timeout: float, **kwargs):
    """otii_tcp_client's own source shows that most Arc calls go through a
    bounded 3s socket timeout, but project.py's start_recording()/
    stop_recording()/get_last_recording() explicitly pass timeout=None ("to
    avoid timeout") and can hang forever if the Otii app is frozen. This
    guarantees the caller regains control after `timeout` seconds even if
    the underlying blocking call never returns. Always runs on AsyncWorker's
    plain asyncio loop (never qasync's), so the standard library's own
    to_thread()/ThreadPoolExecutor/call_soon_threadsafe machinery applies
    here -- no Qt involved, nothing qasync-specific to distrust."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn, *args, **kwargs), timeout=timeout)
    except asyncio.TimeoutError:
        name = getattr(fn, "__name__", str(fn))
        raise OtiiCommsTimeout(f"{name} did not respond within {timeout:.1f}s") from None


# ---------------------------------------------------------------------------
# Settings persistence (cache/settings.json + cache/*_port.txt)
# ---------------------------------------------------------------------------

DEFAULT_PANEL_CONFIG_PATH = PROJECT_ROOT / "panel_config.csv"

DEFAULT_SETTINGS = {
    "otii_exe": r"C:\Users\aniru\AppData\Local\otii3\Otii 3.exe",
    "otii_restart_wait_seconds": 5.0,
    "iv_timeout_seconds": 50.0,
    "otii_call_timeout_seconds": 12.0,
    "summary_csv": str(DEFAULT_SUMMARY_CSV),
    "working_dir": "",  # "" = use the process's current working directory
    "panel_config_path": str(DEFAULT_PANEL_CONFIG_PATH),
    "reference_tolerance_pct": norm_analysis.DEFAULT_TOLERANCE * 100.0,
}

SESSION_CSV_NAME = "LowLightTesting_summary.csv"
PENDING_MEASUREMENT_FILENAME = ".pending_measurement.json"


def load_settings() -> dict:
    path = CACHE_DIR / "settings.json"
    settings = dict(DEFAULT_SETTINGS)
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                saved = json.load(fh)
            if isinstance(saved, dict):
                settings.update(saved)
    except Exception:
        pass
    return settings


def save_settings(settings: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "settings.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)


def load_cached_port(name: str) -> str:
    path = CACHE_DIR / f"{name}_port.txt"
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def save_cached_port(name: str, port: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}_port.txt"
    path.write_text(port, encoding="utf-8")


# ---------------------------------------------------------------------------
# Otii executable auto-discovery.
#
# restart_otii_app() (imported from IV_curve_CURRENT_V3.py) already knows how
# to kill+relaunch Otii given a path, but assumes that path is already valid.
# find_otii_executable() locates it on disk the first time so the operator
# never has to type a path — the result is cached in settings.json afterward.
# ---------------------------------------------------------------------------

_OTII_EXE_NAMES = ("Otii 3.exe", "Otii3.exe")
_OTII_SUBDIR_NAMES = ("otii3", "Otii3", "Otii 3", "Otii")
_OTII_NAME_PATTERN = re.compile(r"^otii\s*3?\.exe$", re.IGNORECASE)
_OTII_SEARCH_MAX_DEPTH = 5


def find_otii_executable() -> str | None:
    """Search common Windows install locations for the Otii 3 executable.
    Cheap, well-known paths are checked first; a depth-limited recursive
    walk of LocalAppData/Program Files is the fallback for less common
    install locations. Returns the first match, or None."""
    env_dirs = [
        os.environ.get("LOCALAPPDATA"),
        os.environ.get("PROGRAMFILES"),
        os.environ.get("PROGRAMFILES(X86)"),
        os.environ.get("PROGRAMW6432"),
    ]

    for base in env_dirs:
        if not base:
            continue
        for sub in _OTII_SUBDIR_NAMES:
            for name in _OTII_EXE_NAMES:
                candidate = Path(base) / sub / name
                if candidate.exists():
                    return str(candidate)

    search_roots = [d for d in env_dirs if d]
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        root_depth = len(root_path.parts)
        for dirpath, dirnames, filenames in os.walk(root_path):
            if len(Path(dirpath).parts) - root_depth >= _OTII_SEARCH_MAX_DEPTH:
                dirnames[:] = []
                continue
            for fname in filenames:
                if _OTII_NAME_PATTERN.match(fname):
                    return str(Path(dirpath) / fname)

    return None


def is_otii_process_running() -> bool:
    """Check for a running Otii 3 process (any of its Electron windows or
    the otii_core.exe backend) via tasklist, so the app doesn't blindly
    relaunch an already-running Otii that just isn't accepting connections
    (e.g. its Automation Server isn't enabled) -- that's a different problem
    a relaunch wouldn't fix."""
    for image_name in ("Otii 3.exe", "Otii3.exe", "otii_core.exe"):
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {image_name}", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            continue
        if image_name.lower() in result.stdout.lower():
            return True
    return False


# ---------------------------------------------------------------------------
# Live sensor pollers — background daemon threads updating thread-safe state.
# The GUI reads this state from a QTimer on the main thread (simpler and
# just as safe as cross-thread Qt signals for a ~2Hz readout).
# ---------------------------------------------------------------------------


class LiveValue:
    def __init__(self):
        self._lock = threading.Lock()
        self.value = None
        self.ts = None
        self.error = None

    def set(self, value) -> None:
        with self._lock:
            self.value = value
            self.ts = time.time()
            self.error = None

    def set_error(self, error: str) -> None:
        with self._lock:
            self.error = error

    def snapshot(self):
        with self._lock:
            return self.value, self.ts, self.error


class TimeSeriesBuffer:
    """Thread-safe (timestamp, value) history for the live timeseries plot,
    independent of RollingIrradiance's short averaging window."""

    def __init__(self, max_age_s: float = TIMESERIES_HISTORY_S):
        self.max_age_s = max_age_s
        self._lock = threading.Lock()
        self._times: deque = deque()
        self._values: deque = deque()

    def add(self, value, ts: float | None = None) -> None:
        ts = time.time() if ts is None else ts
        with self._lock:
            self._times.append(ts)
            self._values.append(value)
            self._prune(ts)

    def _prune(self, now: float) -> None:
        cutoff = now - self.max_age_s
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
            self._values.popleft()

    def snapshot(self):
        with self._lock:
            return list(self._times), list(self._values)


class HistoryIrradiance(RollingIrradiance):
    """RollingIrradiance (imported unchanged from listen.py) plus a longer
    history buffer for the live timeseries plot. serial_reader() only ever
    calls .add(value), so this is a transparent drop-in."""

    def __init__(self, window_seconds: float, history_seconds: float = TIMESERIES_HISTORY_S):
        super().__init__(window_seconds=window_seconds)
        self.history = TimeSeriesBuffer(max_age_s=history_seconds)

    def add(self, value) -> None:
        super().add(value)
        self.history.add(value)


def lux_poll_loop(port: str, live: LiveValue, stop_event: threading.Event, history: TimeSeriesBuffer | None = None) -> None:
    """Background loop polling the Triplett LT68, mirroring the reconnect
    backoff pattern in listen.py's run_periodic_logger()."""
    meter = None
    reconnect_delay = 1.0
    while not stop_event.is_set():
        if meter is None:
            try:
                meter = LT68(port=port, timeout=LUX_TIMEOUT_S)
                _log("info", "lux_connected", port=port)
                reconnect_delay = 1.0
            except Exception as e:
                live.set_error(str(e))
                _log("warn", "lux_connect_failed", port=port, error=str(e))
                if stop_event.wait(min(reconnect_delay, 15.0)):
                    break
                reconnect_delay = min(reconnect_delay * 2.0, 15.0)
                continue

        try:
            value = meter.get_lux(settle=LUX_SETTLE_S)
            live.set(value)
            if history is not None:
                history.add(value)
        except Exception as e:
            live.set_error(str(e))
            _log("warn", "lux_read_failed", error=str(e))
            try:
                meter.close()
            except Exception:
                pass
            meter = None
            if stop_event.wait(min(reconnect_delay, 15.0)):
                break
            reconnect_delay = min(reconnect_delay * 2.0, 15.0)
            continue

        if stop_event.wait(LUX_POLL_INTERVAL_S):
            break

    if meter is not None:
        try:
            meter.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Thread-safe progress bridge.
#
# harvest()'s progress_cb is invoked from the background thread
# call_with_timeout() runs it in — Qt widgets must only be touched
# from the main thread, so the callback emits a signal instead of calling
# QProgressBar.setValue() directly. Qt automatically queues the delivery of
# a signal emitted from a worker thread to a slot owned by a main-thread
# QObject, which is what makes this safe.
# ---------------------------------------------------------------------------


class _ProgressBridge(QObject):
    progress = pyqtSignal(float)


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------


class SettingsDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.otii_exe_edit = QLineEdit(settings.get("otii_exe", ""))
        self.restart_wait_edit = QLineEdit(str(settings.get("otii_restart_wait_seconds", 5.0)))
        self.iv_timeout_edit = QLineEdit(str(settings.get("iv_timeout_seconds", 50.0)))
        self.call_timeout_edit = QLineEdit(str(settings.get("otii_call_timeout_seconds", 12.0)))
        self.summary_csv_edit = QLineEdit(settings.get("summary_csv", str(DEFAULT_SUMMARY_CSV)))
        self.panel_config_edit = QLineEdit(settings.get("panel_config_path", str(DEFAULT_PANEL_CONFIG_PATH)))
        self.tolerance_edit = QLineEdit(str(settings.get("reference_tolerance_pct", 3.0)))

        form.addRow("Otii 3 executable path:", self.otii_exe_edit)
        form.addRow("Otii restart wait (s):", self.restart_wait_edit)
        form.addRow("IV sweep timeout (s):", self.iv_timeout_edit)
        form.addRow("Otii call timeout (s):", self.call_timeout_edit)
        form.addRow("Summary CSV path:", self.summary_csv_edit)
        form.addRow("Panel config file:", self.panel_config_edit)
        form.addRow("1000 W/m² reference tolerance (%):", self.tolerance_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict:
        def _float(edit, default):
            try:
                return float(edit.text().strip())
            except ValueError:
                return default

        return {
            "otii_exe": self.otii_exe_edit.text().strip(),
            "otii_restart_wait_seconds": _float(self.restart_wait_edit, 5.0),
            "iv_timeout_seconds": _float(self.iv_timeout_edit, 50.0),
            "otii_call_timeout_seconds": _float(self.call_timeout_edit, 12.0),
            "summary_csv": self.summary_csv_edit.text().strip() or str(DEFAULT_SUMMARY_CSV),
            "panel_config_path": self.panel_config_edit.text().strip() or str(DEFAULT_PANEL_CONFIG_PATH),
            "reference_tolerance_pct": _float(self.tolerance_edit, 3.0),
        }


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class LowLightApp(QMainWindow):
    MAX_AUTO_RESTARTS = 2
    AUTO_RESTART_WINDOW_S = 5 * 60.0
    _ui_call = pyqtSignal(object)  # marshals a zero-arg callable to the main thread

    def __init__(self):
        super().__init__()
        self.settings = load_settings()

        def _run_marshaled_fn(fn):
            try:
                fn()
            except Exception as e:
                _log("error", "unhandled_on_main_exception", error=str(e))
                try:
                    self.set_status(f"Internal error (caught): {e}", "error")
                except Exception:
                    pass
        self._ui_call.connect(_run_marshaled_fn)
        self._async_worker = AsyncWorker()

        # Otii session state
        self.otii_connection = None
        self.otii_object = None
        self.otii_project = None
        self.otii_devices = []
        self._measuring = False
        self._last_metrics = None
        self._last_recording_info = None
        self._restart_timestamps: list[float] = []
        # Set by "End Sweep Now" (Ctrl+E) to manually cut a running sweep
        # short; checked once per sweep-loop iteration in harvest(). Always
        # cleared at the start of a new measurement so a stale "set" from a
        # previous one can't instantly end the next sweep too.
        self._end_sweep_event = threading.Event()
        # Serializes every call that touches the Otii TCP connection
        # (connect, idle-watchdog ping, measurement) so a stale ping can
        # never race real measurement traffic on the same socket -- see the
        # 2026-09 adversarial review. Safe to construct here even though no
        # event loop is running yet: asyncio.Lock only binds to a loop on
        # first await, and every await of this lock happens on
        # self._async_worker.loop.
        self._otii_lock = asyncio.Lock()
        self._recovering = False

        # Session table state
        self.session_rows: list[dict] = []
        self._next_run_index = 1
        self._last_measurement_window: tuple[float, float] | None = None
        self.working_dir = Path.cwd()

        # Live sensor state
        self.lux_live = LiveValue()
        self.lux_history = TimeSeriesBuffer(max_age_s=TIMESERIES_HISTORY_S)
        self.irr_store = HistoryIrradiance(window_seconds=IRR_WINDOW_S, history_seconds=TIMESERIES_HISTORY_S)
        self._lux_stop = threading.Event()
        self._lux_thread = None
        self._irr_thread = None
        self._sensor_connect_status_pending = False

        # Measurement start/end markers shown as vertical lines on the
        # lux/irradiance timeseries plots. Each entry is a mutable [start, end]
        # pair (end stays None while a measurement is in progress).
        self._measurement_markers: list = []
        self._ping_in_flight = False

        self._build_ui()

        self._progress_bridge = _ProgressBridge()
        self._progress_bridge.progress.connect(self._guard1(self._on_progress_value))

        # Load the panel config before the first Analysis-tab refresh (which
        # _apply_working_dir() below triggers).
        self.panel_config: dict = {}
        self._reload_panel_config()

        # Restore the last-used working folder (if any) and reload whatever
        # session already lives there, for continuity across app launches.
        initial_wd = self.settings.get("working_dir") or str(Path.cwd())
        self._apply_working_dir(Path(initial_wd), persist=False)

        # Auto-detect + auto-connect the sensors on launch so the operator
        # doesn't have to know or pick COM numbers ("set up the listen
        # process automatically"). Manual Refresh/Connect buttons stay
        # available as a fallback/override.
        self._auto_connect_sensors_if_needed()

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(150)
        self._live_timer.timeout.connect(self._guard(self._update_live_readouts))
        self._live_timer.start()

        self._watchdog_timer = QTimer(self)
        self._watchdog_timer.setInterval(7000)
        self._watchdog_timer.timeout.connect(self._guard(self._idle_watchdog_tick))
        self._watchdog_timer.start()

        self._ts_redraw_timer = QTimer(self)
        self._ts_redraw_timer.setInterval(TIMESERIES_REDRAW_MS)
        self._ts_redraw_timer.timeout.connect(self._guard(self._redraw_timeseries))
        self._ts_redraw_timer.start()

    # ------------------------------------------------------------------
    # Shutdown safety
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Refuse to close silently while a measurement is running or while
        a completed-but-unsaved result is sitting at "click Save This Test"
        -- otherwise the X button / Alt+F4 / a Windows shutdown discards a
        real, already-collected measurement with zero warning (2026-09
        adversarial review finding #1)."""
        if self._measuring:
            resp = QMessageBox.question(
                self,
                "Measurement in progress",
                "A measurement is currently running. Closing now will abandon it and "
                "the data will be lost. Close anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        elif self._last_metrics is not None:
            resp = QMessageBox.question(
                self,
                "Unsaved measurement",
                "The last measurement hasn't been saved yet (click \"Save This Test\" first). "
                "Closing now will discard it. Close anyway?",
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

    def _build_ui(self):
        self.setWindowTitle("Low Light IV Tester — Voltaic Systems")
        self.setStyleSheet(_STYLESHEET)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header bar ──────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(52)
        header.setStyleSheet("background:#1a56db; border-bottom:2px solid #1246c0;")
        hlay = QHBoxLayout(header)
        hlay.setContentsMargins(14, 0, 14, 0)
        hlay.setSpacing(14)

        title = QLabel("LOW LIGHT IV TESTER")
        title.setStyleSheet("color:#ffffff; font-size:15pt; font-weight:bold; background:transparent;")
        hlay.addWidget(title)
        hlay.addStretch()

        self.otii_badge = QLabel("● Otii: disconnected")
        self.otii_badge.setStyleSheet("color:#ffcccc; font-size:11pt; font-weight:bold; background:transparent;")
        hlay.addWidget(self.otii_badge)

        self.lux_badge = QLabel("● Lux: --")
        self.lux_badge.setStyleSheet("color:#ffcccc; font-size:11pt; font-weight:bold; background:transparent;")
        hlay.addWidget(self.lux_badge)

        self.irr_badge = QLabel("● Irr: --")
        self.irr_badge.setStyleSheet("color:#ffcccc; font-size:11pt; font-weight:bold; background:transparent;")
        hlay.addWidget(self.irr_badge)

        settings_btn = QPushButton("Settings")
        settings_btn.setStyleSheet(
            "QPushButton { background:#ffffff; color:#1a56db; border:2px solid #ffffff; "
            "border-radius:6px; font-size:10pt; font-weight:bold; min-height:0; padding:4px 10px; }"
            "QPushButton:hover { background:#e8eaed; }"
        )
        settings_btn.clicked.connect(self._guard(self._open_settings_dialog))
        hlay.addWidget(settings_btn)
        root.addWidget(header)

        # ── Tabs: "Measure" (today's whole workflow) + "Analysis" (new) ──
        self._tab_widget = QTabWidget()
        root.addWidget(self._tab_widget, 1)

        # ── Body splitter (upper: plot + controls | lower: table) ───
        # Collapsible (the default) rather than fixed at a hard minimum --
        # on a small/13" screen the whole window is often shorter than the
        # sum of every pane's natural size, and the panes need to actually
        # shrink (by dragging, or automatically) rather than forcing the
        # window bigger than the screen and hiding the table entirely.
        body_split = QSplitter(Qt.Orientation.Vertical)
        self._tab_widget.addTab(body_split, "Measure")

        upper_split = QSplitter(Qt.Orientation.Horizontal)

        # Plot column: IV curve on top, lux/irradiance timeseries below.
        plot_container = QWidget()
        pclay = QVBoxLayout(plot_container)
        pclay.setContentsMargins(4, 4, 4, 4)
        pclay.setSpacing(6)

        fig = Figure(dpi=110, facecolor="#f0f2f5")
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        ax = fig.add_subplot(111)
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (uA)")
        ax.grid(True)
        self.canvas.draw()
        pclay.addWidget(self.canvas, 3)

        ts_fig = Figure(dpi=100, facecolor="#f0f2f5")
        self.ts_canvas = FigureCanvas(ts_fig)
        self.ts_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lux_ts_ax = ts_fig.add_subplot(1, 2, 1)
        self.irr_ts_ax = ts_fig.add_subplot(1, 2, 2)
        self._style_timeseries_axes()
        ts_fig.tight_layout()
        self.ts_canvas.draw()
        pclay.addWidget(self.ts_canvas, 1)

        upper_split.addWidget(plot_container)

        # Controls -- stacking a status badge, prereq banner, live-sensor
        # card, connect button, panel-details form, and results all in one
        # column adds up to more natural height than a 13" laptop screen
        # has room for. Wrapping it in a scroll area (instead of giving it a
        # minimum size) means the splitter can shrink this pane as small as
        # the window needs; the controls just scroll internally rather than
        # forcing the window -- and the table pane below it -- off-screen.
        ctrl = QWidget()
        ctrl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        clay = QVBoxLayout(ctrl)
        clay.setContentsMargins(10, 10, 10, 10)
        clay.setSpacing(8)

        # Status badge
        self.status_badge = QLabel("Ready — connect Otii and the sensors to begin")
        self.status_badge.setWordWrap(True)
        self.status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_badge.setStyleSheet(_STAGE_STYLE["info"])
        clay.addWidget(self.status_badge)

        # Prerequisites banner
        prereq = QLabel(
            "Before connecting: Otii 3 is installed and running, you're signed in, "
            "and an Automation license is reserved for this seat."
        )
        prereq.setWordWrap(True)
        prereq.setStyleSheet(
            "background:#fff4e5; color:#8a5300; border:1px solid #f2a900; "
            "border-radius:5px; padding:6px; font-size:9pt;"
        )
        clay.addWidget(prereq)

        # Progress row (hidden until sweep starts)
        self._prog_row = QWidget()
        prow_lay = QVBoxLayout(self._prog_row)
        prow_lay.setContentsMargins(0, 0, 0, 0)
        prow_lay.setSpacing(2)
        prow_lay.addWidget(QLabel("Sweep progress"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        prow_lay.addWidget(self.progress_bar)
        self.btn_end_sweep = QPushButton("End Sweep Now  (Ctrl+E)")
        self.btn_end_sweep.setProperty("danger", True)
        self.btn_end_sweep.setEnabled(False)
        self.btn_end_sweep.setToolTip(
            "Stop the IV sweep immediately and use whatever data has been "
            "captured so far (Ctrl+E)"
        )
        self.btn_end_sweep.clicked.connect(self._guard(self._end_sweep_now))
        prow_lay.addWidget(self.btn_end_sweep)
        self._prog_row.hide()
        clay.addWidget(self._prog_row)

        clay.addWidget(self._separator())

        # Live sensor card
        clay.addWidget(self._section_label("LIVE SENSORS"))
        sensor_row = QHBoxLayout()
        self.lux_value_lbl = QLabel("-- lx")
        self.lux_value_lbl.setStyleSheet("font-size:16pt; font-weight:bold; color:#1a1a1a;")
        self.irr_value_lbl = QLabel("-- W/m²")
        self.irr_value_lbl.setStyleSheet("font-size:16pt; font-weight:bold; color:#1a1a1a;")
        sensor_row.addWidget(self.lux_value_lbl)
        sensor_row.addWidget(self.irr_value_lbl)
        clay.addLayout(sensor_row)

        sensors_form = QFormLayout()
        self.lux_port_combo = QComboBox()
        self.irr_port_combo = QComboBox()
        self._refresh_port_lists()
        refresh_btn = QPushButton("Refresh Ports")
        refresh_btn.setToolTip("Re-scan COM ports, re-run auto-detect, and connect any newly found sensor")
        refresh_btn.clicked.connect(self._guard(self._auto_connect_sensors_if_needed))
        sensors_form.addRow("Lux meter port:", self.lux_port_combo)
        sensors_form.addRow("Irradiance port:", self.irr_port_combo)
        clay.addLayout(sensors_form)

        sensor_btn_row = QHBoxLayout()
        connect_sensors_btn = QPushButton("Connect Sensors")
        connect_sensors_btn.setProperty("primary", True)
        connect_sensors_btn.clicked.connect(self._guard(self._manual_connect_sensors))
        sensor_btn_row.addWidget(connect_sensors_btn)
        sensor_btn_row.addWidget(refresh_btn)
        clay.addLayout(sensor_btn_row)

        clay.addWidget(self._separator())

        # Otii connect
        clay.addWidget(self._section_label("STEP 1 — Connect to Otii"))
        self.btn_connect = QPushButton("Connect to Otii")
        self.btn_connect.setProperty("primary", True)
        self.btn_connect.clicked.connect(self._guard(self._connect_instrument))
        clay.addWidget(self.btn_connect)

        clay.addWidget(self._separator())

        # Panel inputs
        clay.addWidget(self._section_label("STEP 2 — Panel Details"))
        panel_form = QFormLayout()
        self.panel_name_edit = QLineEdit("P1")
        self.light_meter_edit = QLineEdit("")
        self.si_edit = QLineEdit("1000")
        self.panel_temp_edit = QLineEdit("")
        self.notes_edit = QLineEdit("")
        panel_form.addRow("Panel Name:", self.panel_name_edit)
        panel_form.addRow("Light Meter:", self.light_meter_edit)
        panel_form.addRow("Solar Intensity:", self.si_edit)
        panel_form.addRow("Panel Temp (°C):", self.panel_temp_edit)
        panel_form.addRow("Notes:", self.notes_edit)
        clay.addLayout(panel_form)

        self.btn_run = QPushButton("Run Test  (Ctrl+R)")
        self.btn_run.setProperty("primary", True)
        self.btn_run.setEnabled(False)
        self.btn_run.setToolTip("Run Test (Ctrl+R)")
        self.btn_run.clicked.connect(self._guard(self._start_measurement))
        clay.addWidget(self.btn_run)

        clay.addWidget(self._separator())

        # Results
        clay.addWidget(self._section_label("LAST RESULT"))
        self.results_lbl = QLabel("—")
        self.results_lbl.setWordWrap(True)
        self.results_lbl.setStyleSheet("font-size:11pt;")
        clay.addWidget(self.results_lbl)

        self.btn_save = QPushButton("Save This Test  (Ctrl+S)")
        self.btn_save.setProperty("primary", True)
        self.btn_save.setEnabled(False)
        self.btn_save.setToolTip("Save This Test (Ctrl+S)")
        self.btn_save.clicked.connect(self._guard(self._save_this_test))
        clay.addWidget(self.btn_save)

        run_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        run_shortcut.activated.connect(self._guard(self._run_test_shortcut))
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self._guard(self._save_test_shortcut))
        end_sweep_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
        end_sweep_shortcut.activated.connect(self._guard(self._end_sweep_now))

        clay.addStretch()

        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidget(ctrl)
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFrameShape(QFrame.Shape.NoFrame)
        ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        upper_split.addWidget(ctrl_scroll)
        upper_split.setSizes([650, 380])
        upper_split.setStretchFactor(0, 3)
        upper_split.setStretchFactor(1, 2)
        body_split.addWidget(upper_split)

        # Lower: table + bottom buttons
        bottom = QWidget()
        blay = QVBoxLayout(bottom)
        blay.setContentsMargins(6, 6, 6, 6)
        blay.setSpacing(6)

        wd_row = QHBoxLayout()
        self.working_dir_lbl = QLabel("Working folder: (not set)")
        self.working_dir_lbl.setStyleSheet("color:#444; font-size:9pt;")
        wd_btn = QPushButton("Set Working Folder...")
        wd_btn.setToolTip(
            "Choose the folder this session's IV curve files are saved to. "
            "If it already has saved measurements, they're reloaded."
        )
        wd_btn.clicked.connect(self._guard(self._choose_working_dir))
        wd_row.addWidget(self.working_dir_lbl, 1)
        wd_row.addWidget(wd_btn)
        blay.addLayout(wd_row)

        blay.addWidget(self._section_label("SESSION MEASUREMENTS"))

        self.table = QTableWidget()
        self.table.setColumnCount(len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setRowCount(0)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        for col in range(1, len(_COLUMNS) - 1):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(len(_COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 120)
        copy_sc = QShortcut(QKeySequence.StandardKey.Copy, self.table)
        copy_sc.activated.connect(self._guard(self._copy_table_selection))
        self.table.itemChanged.connect(self._guard1(self._on_table_item_changed))
        blay.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        self.btn_finish = QPushButton("Save All && Finish")
        self.btn_finish.setProperty("primary", True)
        self.btn_finish.clicked.connect(self._guard(self._save_all_and_finish))
        self.btn_reset = QPushButton("Start Over")
        self.btn_reset.setProperty("danger", True)
        self.btn_reset.clicked.connect(self._guard(self._start_over))
        btn_row.addWidget(self.btn_finish, 2)
        btn_row.addWidget(self.btn_reset, 1)
        blay.addLayout(btn_row)

        body_split.addWidget(bottom)
        body_split.setSizes([550, 400])
        body_split.setStretchFactor(0, 3)
        body_split.setStretchFactor(1, 2)

        self._tab_widget.addTab(self._build_analysis_tab(), "Analysis")
        self._tab_widget.addTab(self._build_panel_config_tab(), "Panel Config")

    def _build_analysis_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        self.analysis_status_lbl = QLabel("No data yet.")
        self.analysis_status_lbl.setWordWrap(True)
        self.analysis_status_lbl.setStyleSheet("color:#444; font-size:9pt;")
        self.panel_config_lbl = QLabel("Panel config: (not loaded)")
        self.panel_config_lbl.setStyleSheet("color:#444; font-size:9pt;")
        reload_btn = QPushButton("Reload Panel Config")
        reload_btn.setToolTip("Re-read the panel config file (e.g. after editing it externally) and redraw")
        reload_btn.clicked.connect(self._guard(self._reload_panel_config))
        top_row.addWidget(self.analysis_status_lbl, 1)
        top_row.addWidget(self.panel_config_lbl)
        top_row.addWidget(reload_btn)
        layout.addLayout(top_row)

        # Filters: narrow which panel types/serials are plotted. Applied at
        # display time only (see _refresh_analysis_tab) -- they never change
        # which measurements feed into a group's 1000 W/m^2 reference, only
        # what's drawn, so hiding a panel can't quietly shift another one's
        # normalization.
        filter_row = QHBoxLayout()
        self._filter_serial_list = self._make_filter_list()
        self._filter_celltype_list = self._make_filter_list()
        self._filter_series_list = self._make_filter_list()
        self._filter_parallel_list = self._make_filter_list()
        for label, widget in (
            ("Serial", self._filter_serial_list),
            ("Cell Type", self._filter_celltype_list),
            ("Series", self._filter_series_list),
            ("Parallel", self._filter_parallel_list),
        ):
            col = QVBoxLayout()
            col.setSpacing(2)
            lbl = QLabel(label)
            lbl.setStyleSheet("color:#555; font-size:8pt; font-weight:bold;")
            col.addWidget(lbl)
            col.addWidget(widget)
            filter_row.addLayout(col)
        clear_filters_btn = QPushButton("Clear Filters")
        clear_filters_btn.setToolTip("No selection in a filter list means \"show all\" for that dimension")
        clear_filters_btn.clicked.connect(self._guard(self._clear_analysis_filters))
        filter_row.addWidget(clear_filters_btn, alignment=Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(filter_row)

        analysis_fig = Figure(dpi=100, facecolor="#f0f2f5")
        self.analysis_canvas = FigureCanvas(analysis_fig)
        self.analysis_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        axes = analysis_fig.subplots(2, 2, sharex=True)
        self._norm_wp_ax, self._norm_vp_ax = axes[0]
        self._norm_ip_ax, self._norm_isc_ax = axes[1]
        self._analysis_axes = (self._norm_wp_ax, self._norm_vp_ax, self._norm_ip_ax, self._norm_isc_ax)
        self._style_analysis_axes()
        analysis_fig.tight_layout()
        self.analysis_canvas.draw()
        self._analysis_hover_annotation = None
        self.analysis_canvas.mpl_connect("motion_notify_event", self._guard1(self._on_analysis_hover))

        self.analysis_toolbar = NavigationToolbar2QT(self.analysis_canvas, tab)
        layout.addWidget(self.analysis_toolbar)
        layout.addWidget(self.analysis_canvas, 1)

        return tab

    def _make_filter_list(self) -> QListWidget:
        widget = QListWidget()
        widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        widget.setFixedHeight(70)
        widget.setMinimumWidth(90)
        widget.itemSelectionChanged.connect(self._guard(self._refresh_analysis_tab))
        return widget

    def _build_panel_config_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        intro = QLabel(
            "One row per panel serial prefix (e.g. \"P124\" out of \"P124N042\"). "
            "Panels sharing the same Cell Type + Cells Series + Cells Parallel pool into "
            "one 1000 W/m² reference baseline on the Analysis tab. Edits here save to the "
            "panel config file and refresh the Analysis tab automatically."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#555; font-size:9pt;")
        layout.addWidget(intro)

        self.panel_config_status_lbl = QLabel("")
        self.panel_config_status_lbl.setWordWrap(True)
        self.panel_config_status_lbl.setStyleSheet("color:#444; font-size:9pt;")
        layout.addWidget(self.panel_config_status_lbl)

        self.panel_config_table = QTableWidget()
        self.panel_config_table.setColumnCount(4)
        self.panel_config_table.setHorizontalHeaderLabels(
            ["Serial Prefix", "Cell Type", "Cells Series", "Cells Parallel"]
        )
        hdr = self.panel_config_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        self.panel_config_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.panel_config_table.itemChanged.connect(self._guard(self._on_panel_config_table_changed))
        layout.addWidget(self.panel_config_table, 1)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Row")
        add_btn.clicked.connect(self._guard(self._add_panel_config_row))
        delete_btn = QPushButton("Delete Selected Row(s)")
        delete_btn.setProperty("danger", True)
        delete_btn.clicked.connect(self._guard(self._delete_panel_config_rows))
        reload_btn = QPushButton("Reload From File")
        reload_btn.setToolTip("Discard unsaved table edits and re-read the file from disk")
        reload_btn.clicked.connect(self._guard(self._reload_panel_config))
        btn_row.addWidget(add_btn)
        btn_row.addWidget(delete_btn)
        btn_row.addWidget(reload_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        return tab

    def _separator(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background:#9aa0a6; max-height:1px; border:none;")
        return sep

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#555; font-size:9pt; font-weight:bold; padding:2px 0;")
        return lbl

    def _on_progress_value(self, fraction: float) -> None:
        self.progress_bar.setValue(int(max(0.0, min(1.0, fraction)) * 100))

    def _style_timeseries_axes(self) -> None:
        self.lux_ts_ax.set_title("Lux (live)", fontsize=9)
        self.lux_ts_ax.set_xlabel("Seconds ago", fontsize=8)
        self.lux_ts_ax.set_ylabel("lx", fontsize=8)
        self.lux_ts_ax.grid(True, alpha=0.3)
        self.lux_ts_ax.tick_params(labelsize=7)

        self.irr_ts_ax.set_title("Irradiance (live)", fontsize=9)
        self.irr_ts_ax.set_xlabel("Seconds ago", fontsize=8)
        self.irr_ts_ax.set_ylabel("W/m²", fontsize=8)
        self.irr_ts_ax.grid(True, alpha=0.3)
        self.irr_ts_ax.tick_params(labelsize=7)

    def _style_analysis_axes(self) -> None:
        specs = [
            (self._norm_wp_ax, "Norm Wp (%)"),
            (self._norm_vp_ax, "Norm Vp (%)"),
            (self._norm_ip_ax, "Norm Ip (%)"),
            (self._norm_isc_ax, "Norm Isc (%)"),
        ]
        for ax, title in specs:
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Irradiance (W/m²)", fontsize=8)
            ax.set_ylabel("%", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

    def _reload_panel_config(self) -> None:
        """Re-read the panel config file from disk, repopulate the editor
        table and filter lists, and refresh the Analysis tab. Used on
        startup, after "Reload From File", and after Settings changes."""
        path = self.settings.get("panel_config_path", str(DEFAULT_PANEL_CONFIG_PATH))
        self.panel_config = norm_analysis.load_panel_config(path)
        self.panel_config_lbl.setText(f"Panel config: {path} ({len(self.panel_config)} entries)")
        self._populate_panel_config_table()
        self._populate_analysis_filters()
        self._refresh_analysis_tab()

    def _populate_panel_config_table(self) -> None:
        self.panel_config_table.blockSignals(True)
        self.panel_config_table.setRowCount(0)
        for prefix, entry in sorted(self.panel_config.items()):
            r = self.panel_config_table.rowCount()
            self.panel_config_table.insertRow(r)
            self.panel_config_table.setItem(r, 0, QTableWidgetItem(prefix))
            self.panel_config_table.setItem(r, 1, QTableWidgetItem(entry["cell_type"]))
            self.panel_config_table.setItem(r, 2, QTableWidgetItem(str(entry["cells_series"])))
            self.panel_config_table.setItem(r, 3, QTableWidgetItem(str(entry["cells_parallel"])))
        self.panel_config_table.blockSignals(False)

    def _add_panel_config_row(self) -> None:
        self.panel_config_table.blockSignals(True)
        r = self.panel_config_table.rowCount()
        self.panel_config_table.insertRow(r)
        for c, default in enumerate(("", "", "1", "1")):
            self.panel_config_table.setItem(r, c, QTableWidgetItem(default))
        self.panel_config_table.blockSignals(False)
        self.panel_config_table.editItem(self.panel_config_table.item(r, 0))
        # Left blank (no serial prefix) until the operator actually edits a
        # cell -- _on_panel_config_table_changed() skips prefix-less rows
        # rather than saving/plotting an incomplete one.

    def _delete_panel_config_rows(self) -> None:
        rows = sorted({item.row() for item in self.panel_config_table.selectedItems()}, reverse=True)
        if not rows:
            return
        self.panel_config_table.blockSignals(True)
        for r in rows:
            self.panel_config_table.removeRow(r)
        self.panel_config_table.blockSignals(False)
        self._on_panel_config_table_changed()

    def _on_panel_config_table_changed(self, *_args) -> None:
        """Fires on every completed cell edit (and is called explicitly
        after Add/Delete Row) -- rebuilds self.panel_config from the table's
        current contents, saves it to disk, and refreshes the Analysis tab,
        so "every time a change is made" the graphs stay in sync."""
        config = {}
        invalid_rows = 0

        def _text(r, c):
            item = self.panel_config_table.item(r, c)
            return item.text().strip() if item else ""

        for r in range(self.panel_config_table.rowCount()):
            prefix = _text(r, 0).upper()
            if not prefix:
                continue  # incomplete/new row -- not an error, just not ready yet
            try:
                cells_series = int(float(_text(r, 2)))
                cells_parallel = int(float(_text(r, 3)))
            except ValueError:
                invalid_rows += 1
                continue
            config[prefix] = {
                "cell_type": _text(r, 1),
                "cells_series": cells_series,
                "cells_parallel": cells_parallel,
            }

        self.panel_config = config
        path = self.settings.get("panel_config_path", str(DEFAULT_PANEL_CONFIG_PATH))
        try:
            norm_analysis.save_panel_config(path, config)
            status = f"Saved {len(config)} entries to {path}."
        except Exception as e:
            status = f"Could not save panel config to {path}: {e}"
        if invalid_rows:
            status += f" {invalid_rows} row(s) skipped (Cells Series/Parallel must be numbers)."
        self.panel_config_status_lbl.setText(status)
        self.panel_config_lbl.setText(f"Panel config: {path} ({len(self.panel_config)} entries)")

        self._populate_analysis_filters()
        self._refresh_analysis_tab()

    def _populate_analysis_filters(self) -> None:
        """(Re)populate the 4 filter lists from the currently-loaded panel
        config, preserving each list's current selection where the value
        still exists."""
        serials = sorted(self.panel_config.keys())
        cell_types = sorted({e["cell_type"] for e in self.panel_config.values() if e["cell_type"]})
        series_vals = sorted({str(e["cells_series"]) for e in self.panel_config.values()}, key=lambda s: float(s))
        parallel_vals = sorted({str(e["cells_parallel"]) for e in self.panel_config.values()}, key=lambda s: float(s))

        for widget, values in (
            (self._filter_serial_list, serials),
            (self._filter_celltype_list, cell_types),
            (self._filter_series_list, series_vals),
            (self._filter_parallel_list, parallel_vals),
        ):
            previously_selected = {item.text() for item in widget.selectedItems()}
            widget.blockSignals(True)
            widget.clear()
            for value in values:
                list_item = QListWidgetItem(value)
                widget.addItem(list_item)
                if value in previously_selected:
                    list_item.setSelected(True)
            widget.blockSignals(False)

    def _clear_analysis_filters(self) -> None:
        for widget in (
            self._filter_serial_list, self._filter_celltype_list,
            self._filter_series_list, self._filter_parallel_list,
        ):
            widget.clearSelection()
        self._refresh_analysis_tab()

    def _on_analysis_hover(self, event) -> None:
        """Native hover 'cursor': shows the nearest plotted point's panel
        name, irradiance, and Norm % under the mouse -- no extra dependency
        beyond matplotlib's own event system."""
        ax = event.inaxes
        if ax not in self._analysis_axes or event.x is None:
            if self._analysis_hover_annotation is not None:
                self._analysis_hover_annotation.set_visible(False)
                self.analysis_canvas.draw_idle()
            return

        nearest = None
        nearest_dist = None
        for line in ax.get_lines():
            xdata, ydata = line.get_xdata(), line.get_ydata()
            if len(xdata) == 0:
                continue
            disp = ax.transData.transform(list(zip(xdata, ydata)))
            for (dx, dy), x, y in zip(disp, xdata, ydata):
                dist = (dx - event.x) ** 2 + (dy - event.y) ** 2
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest = (x, y, line.get_label())

        if nearest is None or nearest_dist > 15 ** 2:  # ~15px pick radius
            if self._analysis_hover_annotation is not None:
                self._analysis_hover_annotation.set_visible(False)
                self.analysis_canvas.draw_idle()
            return

        x, y, label = nearest
        text = f"{label}\nIrr: {x:.0f} W/m²\n{y:.1f}%"
        if self._analysis_hover_annotation is None or self._analysis_hover_annotation.axes is not ax:
            if self._analysis_hover_annotation is not None:
                self._analysis_hover_annotation.remove()
            self._analysis_hover_annotation = ax.annotate(
                text, xy=(x, y), xytext=(10, 10), textcoords="offset points",
                fontsize=7, bbox={"boxstyle": "round", "fc": "#ffffe0", "alpha": 0.9},
            )
        else:
            self._analysis_hover_annotation.xy = (x, y)
            self._analysis_hover_annotation.set_text(text)
        self._analysis_hover_annotation.set_visible(True)
        self.analysis_canvas.draw_idle()

    def _refresh_analysis_tab(self) -> None:
        """Recompute Norm Wp/Vp/Ip/Isc from the current working folder's
        summary CSV + the loaded panel config, apply the active filters, and
        redraw the 4 (shared-x-axis) graphs. Always a full recompute from
        the current data (see norm_analysis.compute_norm_dataset) -- cheap
        at this app's data volumes and avoids any incremental-cache
        staleness. Filters only affect what's drawn here, never what feeds
        into a group's reference -- see compute_norm_dataset, which runs on
        the unfiltered dataset."""
        summary_path = self.working_dir / SESSION_CSV_NAME
        rows: list = []
        if summary_path.exists():
            try:
                with summary_path.open("r", newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
            except Exception as e:
                self.analysis_status_lbl.setText(f"Could not read {summary_path}: {e}")
                return

        tolerance = self.settings.get("reference_tolerance_pct", 3.0) / 100.0
        result = norm_analysis.compute_norm_dataset(rows, self.panel_config, tolerance=tolerance)
        groups = result["groups"]

        selected_serials = {item.text() for item in self._filter_serial_list.selectedItems()}
        selected_cell_types = {item.text() for item in self._filter_celltype_list.selectedItems()}
        selected_series = {item.text() for item in self._filter_series_list.selectedItems()}
        selected_parallel = {item.text() for item in self._filter_parallel_list.selectedItems()}

        def group_visible(group_key) -> bool:
            cell_type, series, parallel = group_key
            if selected_cell_types and cell_type not in selected_cell_types:
                return False
            if selected_series and str(series) not in selected_series:
                return False
            if selected_parallel and str(parallel) not in selected_parallel:
                return False
            return True

        axes = self._analysis_axes
        for ax in axes:
            ax.clear()
        self._analysis_hover_annotation = None

        sorted_groups = sorted(groups.items(), key=lambda kv: kv[1]["label"])
        plotted_points = 0
        plotted_groups = 0
        for idx, (group_key, group_data) in enumerate(sorted_groups):
            if not group_visible(group_key):
                continue
            points = group_data["points"]
            if selected_serials:
                points = [
                    p for p in points
                    if norm_analysis.extract_serial_prefix(p["panel_name"]) in selected_serials
                ]
            if not points:
                continue

            plotted_groups += 1
            plotted_points += len(points)
            color = _ANALYSIS_PALETTE[idx % len(_ANALYSIS_PALETTE)]
            irradiance = [p["irradiance"] for p in points]
            for ax, field in zip(axes, ("norm_wp", "norm_vp", "norm_ip", "norm_isc")):
                values = [p[field] for p in points]
                ax.plot(
                    irradiance, values, marker="o", markersize=4, linewidth=1.2,
                    color=color, label=group_data["label"],
                )

        self._style_analysis_axes()
        if plotted_groups:
            self._norm_wp_ax.legend(fontsize=6, loc="best")
        self.analysis_canvas.figure.tight_layout()
        self.analysis_canvas.draw_idle()

        filters_active = bool(selected_serials or selected_cell_types or selected_series or selected_parallel)
        filter_note = " (filtered)" if filters_active else ""
        self.analysis_status_lbl.setText(
            f"{plotted_groups} panel type(s), {plotted_points} point(s) plotted{filter_note} — "
            f"{result['skipped_no_config']} skipped (no panel config match), "
            f"{result['skipped_no_reference']} skipped (no 1000 W/m² reference yet), "
            f"{result['skipped_bad_data']} skipped (missing/bad data)"
        )

    def _redraw_timeseries(self) -> None:
        now = time.time()

        # Drop markers that have scrolled entirely out of the retained history.
        cutoff = now - TIMESERIES_HISTORY_S
        self._measurement_markers = [m for m in self._measurement_markers if (m[1] or m[0]) >= cutoff]

        lux_times, lux_values = self.lux_history.snapshot()
        irr_times, irr_values = self.irr_store.history.snapshot()

        self.lux_ts_ax.clear()
        self.irr_ts_ax.clear()

        if lux_times:
            self.lux_ts_ax.plot([t - now for t in lux_times], lux_values, color="#1a56db", linewidth=1.2)
        if irr_times:
            self.irr_ts_ax.plot([t - now for t in irr_times], irr_values, color="#c53030", linewidth=1.2)

        for ax in (self.lux_ts_ax, self.irr_ts_ax):
            for start, end in self._measurement_markers:
                ax.axvline(start - now, color="#34a853", linestyle="--", linewidth=1.2)
                if end is not None:
                    ax.axvline(end - now, color="#9b1c1c", linestyle="--", linewidth=1.2)

        self._style_timeseries_axes()
        self.ts_canvas.figure.tight_layout()
        self.ts_canvas.draw_idle()

    def _guard(self, fn):
        """Wrap a plain Qt slot method so an unhandled exception is caught,
        logged, and surfaced via set_status() instead of propagating back
        into Qt's C++ call stack. Verified empirically (2026-09 adversarial
        review finding #8): a Python exception escaping a slot connected via
        a direct (same-thread) signal/slot connection -- which is what every
        plain button/table/filter/timer slot in this app uses -- aborts the
        whole process immediately with no traceback at all, regardless of
        any try/except elsewhere in the call stack, and regardless of
        whether an event loop is even running. Overriding
        QApplication.notify() does NOT help (also verified) -- the
        exception has to be caught inside the connected callable itself,
        before control ever returns to C++. This generalizes what
        _launch_task already did narrowly for the handful of slots that
        schedule an async task to every other plain slot.

        Deliberately takes NO parameters itself (not *args/**kwargs) -- also
        verified empirically. PyQt's signal/slot connection logic inspects
        a connected callable's own declared arity to decide how many of the
        signal's arguments to actually pass it (e.g. QPushButton.clicked
        normally trims its bool `checked` argument down to nothing for a
        zero-arg slot); a *args/**kwargs wrapper defeats that -- PyQt sees
        "accepts anything" and passes the signal's arguments through, which
        then blows up calling e.g. _start_measurement(self) with an
        unexpected extra positional argument. functools.wraps() does NOT
        fix this (also verified) -- it only copies cosmetic metadata, not
        the wrapper's actual call signature. Use _guard1() instead for the
        handful of slots that do need the signal's one argument."""
        @functools.wraps(fn)
        def _wrapped():
            try:
                return fn()
            except Exception as e:
                _log("error", "unhandled_slot_exception", slot=getattr(fn, "__name__", str(fn)), error=str(e))
                try:
                    self.set_status(f"Internal error (caught): {e}", "error")
                except Exception:
                    pass
        return _wrapped

    def _guard1(self, fn):
        """Same as _guard(), for the slots that need the signal's one
        argument (QTableWidget.itemChanged's QTableWidgetItem, the progress
        bridge's float, matplotlib's mpl_connect event)."""
        @functools.wraps(fn)
        def _wrapped(arg):
            try:
                return fn(arg)
            except Exception as e:
                _log("error", "unhandled_slot_exception", slot=getattr(fn, "__name__", str(fn)), error=str(e))
                try:
                    self.set_status(f"Internal error (caught): {e}", "error")
                except Exception:
                    pass
        return _wrapped

    def _on_main(self, fn) -> None:
        """Run fn() on the Qt main thread. Safe to call from any thread --
        uses Qt's own native signal/slot queuing (not qasync's), which runs
        fn() immediately if we're already on the main thread, or marshals it
        there otherwise."""
        self._ui_call.emit(fn)

    async def _call_on_main(self, fn, *args, **kwargs):
        """Run fn(*args, **kwargs) on the Qt main thread and await its
        result from a coroutine running on the async worker thread. Use
        this (not call_with_timeout, which runs work OFF the main thread)
        for the handful of operations that must touch Qt widgets or the
        matplotlib canvases directly and whose return value the calling
        coroutine needs (e.g. rendering the IV plot)."""
        loop = asyncio.get_running_loop()
        cf_future: concurrent.futures.Future = concurrent.futures.Future()

        def _runner() -> None:
            try:
                result = fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 -- forwarded to the caller
                cf_future.set_exception(e)
            else:
                cf_future.set_result(result)

        self._on_main(_runner)
        return await asyncio.wrap_future(cf_future, loop=loop)

    def set_status(self, text: str, stage: str = "info") -> None:
        def _apply():
            self.status_badge.setText(text)
            self.status_badge.setStyleSheet(_STAGE_STYLE.get(stage, _STAGE_STYLE["info"]))
        self._on_main(_apply)

    def _set_otii_badge(self, text: str, color: str) -> None:
        def _apply():
            self.otii_badge.setText(text)
            self.otii_badge.setStyleSheet(f"color:{color}; font-size:11pt; font-weight:bold; background:transparent;")
        self._on_main(_apply)

    def _reset_connect_button(self) -> None:
        def _apply():
            self.btn_connect.setEnabled(True)
            self.btn_connect.setText("Connect to Otii")
        self._on_main(_apply)

    def _launch_task(self, coro) -> bool:
        """Schedule a coroutine on the async worker thread without ever
        letting a scheduling failure crash the whole app. PyQt6 aborts the
        process on an unhandled exception raised inside a Qt slot, so this
        has to be caught right here in the synchronous slot. Returns whether
        scheduling succeeded, so callers can undo any UI state (like a
        disabled button) they set in anticipation of the task running."""
        try:
            future = self._async_worker.submit(coro)
        except Exception as e:
            _log("error", "task_schedule_failed", error=str(e))
            try:
                self.set_status(f"Internal error starting task: {e}", "error")
            except Exception:
                pass
            return False

        def _on_done(fut) -> None:
            # AsyncWorker.submit() returns a concurrent.futures.Future whose
            # exception (if any) nobody else ever reads -- unlike an
            # asyncio.Task, a concurrent.futures.Future does NOT log an
            # unretrieved exception on its own, so without this callback a
            # bug inside a task's own code (anything outside its try/except)
            # would be completely silent: no crash, no status message, no
            # log line at all. This callback can fire on any thread.
            try:
                exc = fut.exception()
            except Exception:
                return
            if exc is not None:
                _log("error", "unhandled_task_exception", error=str(exc))
                try:
                    self.set_status(f"Internal error: {exc}", "error")
                except Exception:
                    pass

        future.add_done_callback(_on_done)
        return True

    # ------------------------------------------------------------------
    # Settings dialog
    # ------------------------------------------------------------------

    def _open_settings_dialog(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.settings.update(dlg.values())
            save_settings(self.settings)
            # panel_config_path / reference_tolerance_pct may have changed.
            self._reload_panel_config()

    # ------------------------------------------------------------------
    # Working folder (where this session's IV curve files live)
    # ------------------------------------------------------------------

    def _choose_working_dir(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose Working Folder", str(self.working_dir)
        )
        if not chosen:
            return
        self._apply_working_dir(Path(chosen), persist=True)

    def _apply_working_dir(self, folder: Path, persist: bool) -> None:
        """Switch the session's working folder: everything this session
        saves (per-run PNG/CSV, the summary CSV, Save All & Finish's folder
        open) goes there from now on, and if that folder already holds a
        summary CSV from a previous session, its rows are reloaded into the
        table for continuity."""
        self.working_dir = folder
        self.working_dir_lbl.setText(f"Working folder: {folder}")
        self.settings["summary_csv"] = str(folder / SESSION_CSV_NAME)
        if persist:
            self.settings["working_dir"] = str(folder)
            save_settings(self.settings)

        self.session_rows.clear()
        self.table.setRowCount(0)
        self._next_run_index = 1

        loaded = self._load_session_from_folder(folder)
        if loaded:
            self.set_status(f"Resumed session in {folder} — {loaded} measurement(s) reloaded.", "success")
        else:
            self.set_status(f"Working folder set to {folder}.", "info")

        # A leftover pending-measurement file here means a previous session
        # crashed/closed after the mandatory temp prompt but before Save
        # This Test -- surface it so that measurement's data isn't silently
        # forgotten (2026-09 adversarial review finding #5). Overrides the
        # status set just above since it's the more urgent thing to know.
        self._check_pending_measurement_file(folder)

        self._refresh_analysis_tab()

    _NUMERIC_ROW_FIELDS = (
        "Wp", "Voc", "Isc", "Vp", "Ip",
        "lux_measured", "irradiance_measured_live", "irradiance_live_samples",
        "lux_stdev", "irradiance_stdev", "lux_3sigma_pct", "irradiance_3sigma_pct",
    )

    def _load_session_from_folder(self, folder: Path) -> int:
        """Read folder's summary CSV (if any) and repopulate the session
        table from it -- "if there are already measurements saved in that
        folder, that session is reloaded" (2026-09 feedback). Returns how
        many rows were loaded."""
        summary_path = folder / SESSION_CSV_NAME
        if not summary_path.exists():
            return 0

        loaded = 0
        try:
            with summary_path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for raw_row in reader:
                    row = dict(raw_row)
                    for key in self._NUMERIC_ROW_FIELDS:
                        value = row.get(key, "")
                        if value in (None, ""):
                            row[key] = ""
                            continue
                        try:
                            row[key] = float(value)
                        except ValueError:
                            row[key] = ""
                    self.session_rows.append(row)
                    self._append_table_row(row)
                    loaded += 1
        except Exception as e:
            _log("error", "session_reload_failed", folder=str(folder), error=str(e))
            self.set_status(f"Could not reload session from {folder}: {e}", "warning")
            return loaded

        try:
            self._next_run_index = read_max_run_index(str(summary_path)) + 1
        except Exception:
            self._next_run_index = loaded + 1

        return loaded

    # ------------------------------------------------------------------
    # Sensor ports
    # ------------------------------------------------------------------

    def _refresh_port_lists(self) -> dict:
        """Repopulate both port combo boxes (device + description) and
        auto-select the best guess for each: a live auto-detected match
        first, falling back to the last-used cached port. Returns the
        detect_sensor_ports() result so callers can decide whether to
        auto-connect."""
        ports = list(serial.tools.list_ports.comports())
        detected = detect_sensor_ports()
        cached_lux = load_cached_port("lux")
        cached_irr = load_cached_port("irr")

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
                # Not currently enumerated (device unplugged) — still show it
                # so the operator can see what was last used, findData() below
                # will still resolve it since it's now in the combo either way.
                combo.addItem(f"{cached} (not currently connected)", cached)

            preferred = detected_port or cached
            if preferred:
                idx = combo.findData(preferred)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        return detected

    def _auto_connect_sensors_if_needed(self):
        """Auto-detect the sensor COM ports by their USB bridge chip and
        start the background polling ("listen") threads automatically,
        without requiring the operator to know or pick COM numbers. Only
        touches a sensor that isn't already connected, so this is safe to
        call repeatedly (e.g. every time Refresh Ports is clicked)."""
        detected = self._refresh_port_lists()
        lux_running = self._lux_thread is not None and self._lux_thread.is_alive()
        irr_running = self._irr_thread is not None and self._irr_thread.is_alive()

        found = [name for name in ("lux", "irr") if detected[name] and not (lux_running if name == "lux" else irr_running)]
        if found:
            names = " & ".join({"lux": "lux meter", "irr": "irradiance sensor"}[n] for n in found)
            self.set_status(f"Auto-detected {names} — connecting…", "info")

        if (detected["lux"] and not lux_running) or (detected["irr"] and not irr_running):
            self._connect_sensors()
        elif not detected["lux"] and not detected["irr"] and not lux_running and not irr_running:
            self.set_status(
                "Could not auto-detect the lux meter (CP210x) or irradiance sensor (CH9102) — "
                "check they're plugged in, or pick their ports manually below.",
                "warning",
            )

    def _manual_connect_sensors(self):
        self.set_status("Connecting sensor(s)…", "info")
        self._connect_sensors()

    def _connect_sensors(self):
        lux_port = self.lux_port_combo.currentData() or self.lux_port_combo.currentText().strip()
        irr_port = self.irr_port_combo.currentData() or self.irr_port_combo.currentText().strip()
        started_any = False

        if lux_port:
            save_cached_port("lux", lux_port)
            if self._lux_thread is None or not self._lux_thread.is_alive():
                self._lux_stop.clear()
                self._lux_thread = threading.Thread(
                    target=lux_poll_loop,
                    args=(lux_port, self.lux_live, self._lux_stop, self.lux_history),
                    daemon=True,
                )
                self._lux_thread.start()
                started_any = True

        if irr_port:
            save_cached_port("irr", irr_port)
            if self._irr_thread is None or not self._irr_thread.is_alive():
                self._irr_thread = threading.Thread(
                    target=serial_reader,
                    args=(irr_port, IRR_BAUD_DEFAULT, self.irr_store, 10**9, 1.0, 2.0, 30.0, _log),
                    daemon=True,
                )
                self._irr_thread.start()
                started_any = True

        if started_any:
            # Resolved once _update_live_readouts() sees both requested
            # sensors actually reporting live data — see point 2 of the
            # 2026-09 feedback: the "connecting…" banner used to get stuck.
            self._sensor_connect_status_pending = True

        if not lux_port and not irr_port:
            self.set_status("Pick at least one sensor COM port before connecting.", "warning")

    def _update_live_readouts(self):
        lux_value, lux_ts, lux_err = self.lux_live.snapshot()
        now = time.time()
        lux_is_live = lux_value is not None and (lux_ts is None or now - lux_ts <= LIVE_STALE_S)
        if lux_is_live:
            self.lux_value_lbl.setText(f"{lux_value:.1f} lx")
            self.lux_badge.setText("● Lux: live")
            self.lux_badge.setStyleSheet("color:#a6ffcc; font-size:11pt; font-weight:bold; background:transparent;")
        elif lux_err:
            self.lux_value_lbl.setText("-- lx (error)")
            self.lux_badge.setText("● Lux: error")
            self.lux_badge.setStyleSheet("color:#ffcccc; font-size:11pt; font-weight:bold; background:transparent;")
        else:
            self.lux_value_lbl.setText("-- lx")
            self.lux_badge.setText("● Lux: --")
            self.lux_badge.setStyleSheet("color:#ffe6a6; font-size:11pt; font-weight:bold; background:transparent;")

        # The live number tracks the single most recent raw sample (like
        # Lux above), NOT irr_store.average()'s 10s rolling mean -- that
        # average is still exactly right for a *saved* measurement (see
        # _windowed_stats), but for the live readout it made the number lag
        # a real step change by several seconds while the graph next to it
        # (plotted from the same raw samples) showed it instantly. Video
        # review (2026-09) confirmed irradiance lagging visibly behind its
        # own graph while lux -- already latest-sample-based -- did not.
        irr_value, irr_ts = self.irr_store.latest()
        irr_is_live = irr_value is not None and (irr_ts is None or now - irr_ts <= LIVE_STALE_S)
        if irr_is_live:
            self.irr_value_lbl.setText(f"{irr_value:.4f} W/m²")
            self.irr_badge.setText("● Irr: live")
            self.irr_badge.setStyleSheet("color:#a6ffcc; font-size:11pt; font-weight:bold; background:transparent;")
        else:
            self.irr_value_lbl.setText("-- W/m²")
            self.irr_badge.setText("● Irr: --")
            self.irr_badge.setStyleSheet("color:#ffe6a6; font-size:11pt; font-weight:bold; background:transparent;")

        # Resolve the "connecting…" status banner once whichever sensors were
        # actually requested prove they're live — instead of leaving it stuck
        # on "connecting…" forever (2026-09 feedback point 2).
        if self._sensor_connect_status_pending:
            lux_wanted = bool(self.lux_port_combo.currentData())
            irr_wanted = bool(self.irr_port_combo.currentData())
            lux_ready = (not lux_wanted) or lux_is_live
            irr_ready = (not irr_wanted) or irr_is_live
            if lux_ready and irr_ready:
                self._sensor_connect_status_pending = False
                self.set_status("Sensors connected — connect Otii to begin", "success")

    # ------------------------------------------------------------------
    # Otii connection
    # ------------------------------------------------------------------

    def _connect_instrument(self):
        # Disable synchronously, before scheduling -- a plain
        # asyncio.ensure_future() doesn't run any of the coroutine's body
        # (including its own setEnabled(False)) until the next loop
        # iteration, leaving a brief window where a rapid second click
        # could schedule a second overlapping connect task.
        self.btn_connect.setEnabled(False)
        if not self._launch_task(self._async_connect_instrument()):
            self.btn_connect.setEnabled(True)

    async def _resolve_otii_exe_path(self) -> str | None:
        """Return a valid Otii 3 executable path, auto-locating and caching
        one if the configured path doesn't exist on disk. Used both on first
        connect (point 3 of the 2026-09 feedback) and by the comms-lost
        auto-restart flow."""
        exe_path = self.settings.get("otii_exe", "")
        if exe_path and Path(exe_path).exists():
            return exe_path

        self.set_status("Otii not found at the configured path — searching for it…", "info")
        try:
            found = await call_with_timeout(find_otii_executable, timeout=30.0)
        except OtiiCommsTimeout:
            return None
        if found is None:
            return None

        self.settings["otii_exe"] = found
        save_settings(self.settings)
        self.set_status(f"Found Otii at {found} (cached for next time).", "info")
        return found

    async def _ensure_otii_running(self) -> bool:
        """Auto-locate (if needed) and launch Otii when it doesn't seem to
        be running at all yet. Returns True once launched (not once actually
        ready — the caller still retries the connect afterward)."""
        exe_path = await self._resolve_otii_exe_path()
        if exe_path is None:
            self.set_status(
                "Could not locate Otii 3.exe automatically. Set the path in Settings, then Connect again.",
                "error",
            )
            return False

        try:
            already_running = await call_with_timeout(is_otii_process_running, timeout=10.0)
        except OtiiCommsTimeout:
            already_running = False

        if already_running:
            # It's running but not accepting connections -- relaunching
            # wouldn't help and would just pile up extra instances. This is
            # almost always an Otii-side config issue (Automation Server not
            # enabled, or no Automation license reserved), not something this
            # app can fix by itself.
            _log("warn", "otii_running_but_unreachable", exe=exe_path)
            self.set_status(
                "Otii 3 is already running but isn't accepting connections on port 1905. "
                "In Otii, check that the Automation Server is enabled and an Automation "
                "license is reserved for this seat, then click Connect to Otii again.",
                "error",
            )
            return False

        self.set_status("Otii doesn't seem to be running — launching it…", "info")
        try:
            await call_with_timeout(
                subprocess.Popen,
                [exe_path],
                timeout=10.0,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            msg = str(e)
            self.set_status(f"Failed to launch Otii: {msg}", "error")
            return False

        await asyncio.sleep(self.settings["otii_restart_wait_seconds"])
        return True

    def _close_otii_connection(self) -> None:
        """Close the previous Otii TCP socket (if any) before replacing it
        with a fresh one on every connect/reconnect/auto-restart-retry --
        otherwise each cycle over a long session leaks a socket that's
        never explicitly closed (2026-09 adversarial review finding #6).
        Must only be called while holding self._otii_lock, since it mutates
        connection state shared with the ping/measurement paths."""
        old = self.otii_connection
        if old is None:
            return
        try:
            old.close_connection()
        except Exception:
            pass

    async def _async_connect_instrument(self):
        self.set_status("Connecting to Otii…", "info")
        # Any Otii action supersedes the (possibly still-pending) sensor
        # "connecting…" banner so the two don't fight over the status text.
        self._sensor_connect_status_pending = False
        self._set_otii_badge("● Otii: connecting…", "#ffe6a6")

        # Holds the same lock a measurement/ping would hold while touching
        # the connection, so a connect/reconnect can never interleave with
        # either of those on the same socket (2026-09 adversarial review).
        async with self._otii_lock:
            try:
                connection, otii_object, active_proj, devices = await call_with_timeout(
                    connect_otii_session, timeout=self.settings["otii_call_timeout_seconds"]
                )
            except OtiiCommsTimeout as e:
                self._show_otii_error(str(e))
                return
            except OSError as e:
                # Connection actively refused/unreachable -- Otii almost
                # certainly isn't running. Auto-locate + launch it, then retry
                # the connect once (point 3 of the 2026-09 feedback).
                launched = await self._ensure_otii_running()
                if not launched:
                    # _ensure_otii_running() already set a specific status
                    # message (e.g. "already running but unreachable" vs.
                    # "couldn't be located") -- don't overwrite it here.
                    self._show_otii_error(str(e), set_status_text=False)
                    return
                try:
                    connection, otii_object, active_proj, devices = await call_with_timeout(
                        connect_otii_session, timeout=self.settings["otii_call_timeout_seconds"]
                    )
                except Exception as e2:
                    self._show_otii_error(str(e2))
                    return
            except Exception as e:
                self._show_otii_error(str(e))
                return

            self._close_otii_connection()
            self.otii_connection = connection
            self.otii_object = otii_object
            self.otii_project = active_proj
            self.otii_devices = devices
            self._restart_timestamps.clear()

        self._set_otii_badge(f"● Otii: connected ({len(devices)} Arc)", "#a6ffcc")
        self.set_status("Otii connected — fill in panel details and click Run Test", "success")

        def _enable_ui():
            self.btn_connect.setEnabled(True)
            self.btn_connect.setText("Reconnect to Otii")
            self.btn_run.setEnabled(True)
        self._on_main(_enable_ui)

    def _show_otii_error(self, message: str, set_status_text: bool = True):
        self._set_otii_badge("● Otii: error", "#ffcccc")
        if set_status_text:
            self.set_status(
                f"Otii connection failed: {message}\n"
                "Check: Otii 3 running, signed in, Automation license reserved, Arc plugged in.",
                "error",
            )
        self._on_main(lambda: self.btn_connect.setEnabled(True))

    # ------------------------------------------------------------------
    # Idle latency watchdog
    # ------------------------------------------------------------------

    def _idle_watchdog_tick(self):
        # Guard against a ping that's still in flight when the next 7s tick
        # fires (e.g. Otii responding slowly) -- overlapping pings would
        # schedule two tasks back-to-back.
        if self.otii_object is None or self._measuring or self._ping_in_flight:
            return
        self._ping_in_flight = True
        if not self._launch_task(self._ping_otii()):
            self._ping_in_flight = False

    async def _ping_otii(self):
        # If a measurement or a connect/reconnect is already using the Otii
        # connection, skip this tick rather than queuing behind it -- a
        # ping that then judges a now-stale result once it finally gets the
        # lock is exactly what let a stale idle-health-check ping
        # force-restart Otii out from under an active measurement (2026-09
        # adversarial review finding #3). It'll be checked again next tick.
        if self._otii_lock.locked():
            self._ping_in_flight = False
            return

        started = time.monotonic()
        try:
            async with self._otii_lock:
                await call_with_timeout(
                    self.otii_object.get_devices, timeout=self.settings["otii_call_timeout_seconds"]
                )
        except OtiiCommsTimeout:
            await self._handle_otii_comms_lost("Otii stopped responding (idle health check)")
            return
        except Exception:
            return
        finally:
            self._ping_in_flight = False

        latency = time.monotonic() - started
        if latency > self.settings["otii_call_timeout_seconds"] / 2:
            self._set_otii_badge(f"● Otii: slow ({latency:.1f}s)", "#ffe6a6")
        else:
            self._set_otii_badge("● Otii: connected", "#a6ffcc")

    # ------------------------------------------------------------------
    # Comms-lost / auto-restart recovery
    # ------------------------------------------------------------------

    async def _handle_otii_comms_lost(self, reason: str):
        if self._recovering:
            # Already mid-recovery from a previous comms-lost event (e.g.
            # this got triggered from both the idle-watchdog ping and a
            # measurement's own timeout landing close together) -- don't
            # pile up a second concurrent taskkill/relaunch sequence
            # (2026-09 adversarial review finding #7).
            _log("warn", "comms_lost_reentry_skipped", reason=reason)
            return
        self._recovering = True
        try:
            self._measuring = False
            self.otii_object = None
            self.otii_project = None
            self.otii_devices = []

            def _lost_ui():
                self._prog_row.hide()
                self.btn_run.setEnabled(False)
                self.btn_save.setEnabled(False)
            self._on_main(_lost_ui)
            self._set_otii_badge("● Otii: lost", "#ffcccc")

            now = time.monotonic()
            self._restart_timestamps = [t for t in self._restart_timestamps if now - t < self.AUTO_RESTART_WINDOW_S]

            if len(self._restart_timestamps) >= self.MAX_AUTO_RESTARTS:
                self.set_status(
                    f"{reason}. Otii has been auto-restarted {self.MAX_AUTO_RESTARTS} times recently — "
                    "click Connect to Otii manually after checking the app.",
                    "error",
                )
                self._reset_connect_button()
                return

            exe_path = await self._resolve_otii_exe_path()
            if exe_path is None:
                self.set_status(
                    f"{reason}. Could not locate Otii 3.exe to restart it — set the path in Settings.",
                    "error",
                )
                self._reset_connect_button()
                return

            self._restart_timestamps.append(now)
            self.set_status(f"{reason}. Restarting Otii app…", "warning")
            try:
                restart_wait = self.settings["otii_restart_wait_seconds"]
                await call_with_timeout(
                    restart_otii_app, exe_path, restart_wait, timeout=restart_wait + 20.0
                )
            except Exception as e:
                msg = str(e)
                self.set_status(f"Automatic Otii restart failed: {msg}. Restart it manually, then Connect.", "error")
                self._reset_connect_button()
                return

            self.set_status(
                "Otii restarted. Make sure the Arc Pro is connected/powered, then click Connect to Otii.",
                "warning",
            )
            self._reset_connect_button()
        finally:
            self._recovering = False

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def _run_test_shortcut(self):
        # Mirror the button's own enabled state so Ctrl+R can't start a
        # second overlapping measurement or fire before Otii is connected.
        if self.btn_run.isEnabled():
            self._start_measurement()

    def _save_test_shortcut(self):
        if self.btn_save.isEnabled():
            self._save_this_test()

    def _end_sweep_now(self):
        """Manually end the IV sweep in progress -- e.g. if it's taking too
        long, or appears stuck despite the progress bar showing it's
        nearly/fully done. Whatever the sweep has captured so far is
        treated as valid data (explicit operator instruction) and proceeds
        through the exact same render/save path a natural completion
        would. Only does anything while a sweep is actually running (the
        button/shortcut are only enabled during that window)."""
        if not self.btn_end_sweep.isEnabled():
            return
        self.btn_end_sweep.setEnabled(False)
        self._end_sweep_event.set()
        self.set_status("Ending the sweep now — using the data captured so far…", "info")

    def _start_measurement(self):
        # Disabling Connect too closes the window where a "Reconnect to
        # Otii" click could otherwise run concurrently with this
        # measurement's own Otii calls (2026-09 adversarial review finding
        # #3/#6) -- belt-and-suspenders alongside self._otii_lock, which
        # covers it even if some future code path forgets this button.
        self.btn_run.setEnabled(False)
        self.btn_connect.setEnabled(False)
        if not self._launch_task(self._async_start_measurement()):
            self.btn_run.setEnabled(True)
            self.btn_connect.setEnabled(True)

    def _read_panel_inputs(self):
        return (
            self.panel_name_edit.text().strip(),
            self.light_meter_edit.text().strip(),
            self.si_edit.text().strip(),
        )

    def _prompt_for_temp_blocking(self, panel_name: str) -> str:
        """Modal, inescapable prompt for the panel temperature, shown right
        after every successful measurement. No Cancel/skip path on
        purpose -- an invalid entry, an empty entry, or dismissing the
        dialog (Esc / the window's own close button) all just re-prompt
        until a numeric value is entered. Runs on the main thread (called
        via _call_on_main from the async worker thread)."""
        while True:
            text, ok = QInputDialog.getText(
                self,
                "Panel Temperature Required",
                f"Enter the panel temperature (°C) for \"{panel_name}\" before continuing.\n"
                "This measurement can't be meaningfully compared without it.",
            )
            text = text.strip()
            if not ok or not text:
                continue
            try:
                float(text)
            except ValueError:
                QMessageBox.warning(self, "Invalid Temperature", "Enter a numeric temperature (e.g. 24.7).")
                continue
            return text

    async def _async_start_measurement(self):
        # QLineEdit.text() must be read on the main thread; grab everything
        # this coroutine needs up front so nothing later has to touch a
        # widget directly from the async worker thread.
        panel_name, light_meter, si_text = await self._call_on_main(self._read_panel_inputs)

        # _start_measurement() already disabled btn_run synchronously before
        # scheduling this task, so every early-return path here must
        # re-enable it -- these two used to run before the button was ever
        # touched, but that's no longer true.
        def _reenable_run_and_connect():
            self.btn_run.setEnabled(True)
            self.btn_connect.setEnabled(True)

        if self.otii_project is None or not self.otii_devices:
            self.set_status("No instrument connected", "error")
            self._on_main(_reenable_run_and_connect)
            return
        if not panel_name:
            self.set_status("Panel Name is required before running a test", "warning")
            self._on_main(_reenable_run_and_connect)
            return

        self._measuring = True
        self._end_sweep_event.clear()

        def _begin_ui():
            self.btn_run.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.btn_end_sweep.setEnabled(False)  # only turned on once the sweep itself starts
            self._prog_row.show()
            self.progress_bar.setValue(0)
        self._on_main(_begin_ui)
        self.set_status("Measuring short-circuit current (Isc)…", "info")

        marker = [time.time(), None]
        self._measurement_markers.append(marker)
        self._on_main(self._redraw_timeseries)

        call_timeout = self.settings["otii_call_timeout_seconds"]
        sweep_timeout = self.settings["iv_timeout_seconds"]

        try:
            # Holds the same lock a connect/reconnect or an idle-watchdog
            # ping would hold before touching the Otii connection, so
            # neither can interleave with this measurement's own traffic on
            # the same TCP socket (2026-09 adversarial review finding #3).
            # Only wraps the actual Otii calls -- rendering/saving below
            # doesn't need the connection, so it's done outside the lock.
            async with self._otii_lock:
                isc_measured = await call_with_timeout(
                    short_circuit, self.otii_project, self.otii_devices, timeout=max(call_timeout, 15.0)
                )
                if isc_measured <= 0:
                    raise RuntimeError(f"Invalid Isc measured: {isc_measured}")

                current_step = (isc_measured / 150) * 1e6
                if current_step <= 0:
                    raise RuntimeError(f"Invalid current step derived from Isc: {current_step}")

                self.set_status("Sweeping IV curve…", "info")
                self._on_main(lambda: self.btn_end_sweep.setEnabled(True))

                def _on_progress(fraction: float):
                    # Called from the worker thread harvest() runs in —
                    # must not touch Qt widgets directly, so route through
                    # a signal.
                    self._progress_bridge.progress.emit(fraction)

                def _on_live_data(mv, mc):
                    # Also called from harvest()'s own thread -- it fetches
                    # this data itself, on its own single connection,
                    # throttled to ~once/second (see live_data_interval_s
                    # below), so this can never race harvest()'s own
                    # in-flight Otii requests. Best-effort: any failure here
                    # must never surface past this function, since it's a
                    # live preview, not the measurement.
                    try:
                        mv_arr = np.asarray(mv, dtype=float)
                        mc_arr = np.asarray(mc, dtype=float)
                        window, skip = 25, 10
                        if len(mv_arr) <= window + skip:
                            return
                        mv_smooth = moving_average(mv_arr, window)[:-skip]
                        mc_smooth = -1 * moving_average(mc_arr, window)[:-skip]

                        def _draw_live():
                            fig = self.canvas.figure
                            fig.clf()
                            ax = fig.add_subplot(111)
                            ax.plot(mv_smooth, 1e6 * mc_smooth, color="#1a56db", linewidth=1.5)
                            ax.set_xlabel("Voltage (V)")
                            ax.set_ylabel("Current (uA)")
                            ax.set_title("Sweeping IV curve… (live)")
                            ax.grid(True)
                            self.canvas.draw()

                        self._on_main(_draw_live)
                    except Exception:
                        pass

                try:
                    recording = await call_with_timeout(
                        harvest,
                        self.otii_project,
                        self.otii_devices,
                        current_step,
                        sweep_timeout,
                        _on_progress,
                        _on_live_data,
                        timeout=sweep_timeout + max(call_timeout, 15.0),
                        stop_event=self._end_sweep_event,
                    )
                finally:
                    self._on_main(lambda: self.btn_end_sweep.setEnabled(False))

                my_arc = self.otii_devices[0]

                # The network fetch stays inside the lock (and off the main
                # thread, via call_with_timeout) -- this is the same
                # unbounded-Otii-hang risk as start_recording/stop_recording,
                # previously run straight on the Qt main thread with no
                # timeout at all inside plot_iv_curve() (2026-09 adversarial
                # review finding #2). Only the local render+save below
                # (no network, just numpy/matplotlib/disk) goes on the main
                # thread, and only after the data's already safely in hand.
                mv_data_dict, mc_data_dict, mp_data_dict = await call_with_timeout(
                    fetch_recording_channels, recording, my_arc,
                    timeout=max(call_timeout, 15.0),
                )

            def _render_and_draw():
                result = render_iv_curve(
                    mv_data_dict, mc_data_dict, mp_data_dict, panel_name, si_text, light_meter,
                    show_plot=False, fig=self.canvas.figure, out_dir=str(self.working_dir),
                )
                self.canvas.draw()
                return result

            # Rendered and drawn together, atomically, on the main thread --
            # both touch the same matplotlib Figure/Qt canvas, so keeping
            # them on one thread avoids any chance of interleaving with an
            # unrelated repaint (e.g. a window resize) mid-render.
            metrics = await self._call_on_main(_render_and_draw)

        except OtiiCommsTimeout as e:
            marker[1] = time.time()
            self._on_main(self._redraw_timeseries)
            self._on_main(lambda: self._prog_row.hide())
            self._measuring = False
            await self._handle_otii_comms_lost(f"Measurement abandoned — {e}")
            return
        except Exception as e:
            msg = str(e)
            marker[1] = time.time()
            self._on_main(self._redraw_timeseries)
            self._on_main(lambda: self._prog_row.hide())
            self._measuring = False
            self.set_status(f"Measurement failed: {msg}", "error")
            self._on_main(_reenable_run_and_connect)
            return

        marker[1] = time.time()
        self._on_main(self._redraw_timeseries)
        self._measuring = False
        self._last_metrics = metrics
        self._last_measurement_window = (marker[0], marker[1])
        self._on_main(lambda: self.btn_connect.setEnabled(True))

        # Blocking, inescapable prompt: a measurement without a recorded
        # panel temperature "makes the data point useless" (explicit
        # operator requirement), so this happens immediately after every
        # successful sweep, before anything else -- including Save This
        # Test and starting the next Run Test -- can proceed. QInputDialog
        # is application-modal, so the whole window is genuinely
        # unresponsive to anything else while it's up.
        temp_value = await self._call_on_main(self._prompt_for_temp_blocking, panel_name)
        self._on_main(lambda: self.panel_temp_edit.setText(temp_value))

        # Best-effort crash-recovery net: written the moment every field a
        # summary-CSV row needs actually exists, so a crash or a forgotten
        # "Save This Test" click doesn't silently discard a measurement that
        # already went through the mandatory temp prompt (2026-09
        # adversarial review finding #5). Cleared again once Save This Test
        # actually succeeds -- see _save_this_test().
        lux_value, _, _ = self.lux_live.snapshot()
        irr_value, irr_count = self.irr_store.average()
        pending_snapshot = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "lux": lux_value if lux_value is not None else 0.0,
            "irradiance": irr_value if irr_value is not None else 0.0,
            "samples": irr_count,
        }
        self._write_pending_measurement_snapshot(panel_name, light_meter, si_text, temp_value, metrics, pending_snapshot)

        def _finish_ui():
            self._prog_row.hide()
            self.results_lbl.setText(
                f"Voc: {metrics['Voc']:.4g} V   Isc: {metrics['Isc']:.4g} A\n"
                f"Vp: {metrics['Vp']:.4g} V   Ip: {metrics['Ip']:.4g} A\n"
                f"Pmax: {metrics['Wp']:.4g} W"
            )
            self.btn_run.setEnabled(True)
            self.btn_save.setEnabled(True)
        self._on_main(_finish_ui)
        self.set_status("Measurement complete — review the plot, then Save This Test", "success")

    # ------------------------------------------------------------------
    # Pending-measurement crash-recovery snapshot
    # ------------------------------------------------------------------

    def _pending_measurement_path(self) -> Path:
        return self.working_dir / PENDING_MEASUREMENT_FILENAME

    def _write_pending_measurement_snapshot(self, panel_name, light_meter, si_text, temp_value, metrics, snapshot) -> None:
        """Best-effort only -- must never break the measurement flow if the
        working folder is briefly unwritable (e.g. OneDrive contention)."""
        try:
            payload = {
                "panel_name": panel_name,
                "light_meter": light_meter,
                "irradiance_gui_input": si_text,
                "panel_temp_c": temp_value,
                "metrics": metrics,
                "snapshot": snapshot,
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._pending_measurement_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            _log("warn", "pending_measurement_write_failed", error=str(e))

    def _clear_pending_measurement_snapshot(self) -> None:
        try:
            path = self._pending_measurement_path()
            if path.exists():
                path.unlink()
        except Exception:
            pass

    def _check_pending_measurement_file(self, folder: Path) -> None:
        """Called whenever the working folder is (re)opened. A leftover
        file here means Save This Test never confirmed a previous
        measurement -- its raw IV curve PNG/CSV are safe (written earlier,
        during the measurement itself), but the enriched summary row
        (temp, notes, lux/irr stats) never made it into the summary CSV."""
        path = folder / PENDING_MEASUREMENT_FILENAME
        if not path.exists():
            return
        panel, written_at = "?", "?"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            panel = data.get("panel_name", "?")
            written_at = data.get("written_at", "?")
        except Exception:
            pass
        self.set_status(
            f"Found an unsaved measurement from a previous session ({panel}, {written_at}): {path} — "
            "it was never confirmed with Save This Test, so it's not in the summary CSV yet. "
            "Check the file for its recorded values, then delete it once handled.",
            "warning",
        )

    # ------------------------------------------------------------------
    # Session table
    # ------------------------------------------------------------------

    def _windowed_values(self, history: TimeSeriesBuffer):
        """Raw samples from a live-sensor history that fall within the
        just-completed measurement's window (self._last_measurement_window,
        set alongside self._last_metrics). Shared by the stdev/mean helpers
        below so both are computed from the exact same sample set."""
        if self._last_measurement_window is None:
            return None
        start, end = self._last_measurement_window
        times, values = history.snapshot()
        return [v for t, v in zip(times, values) if start <= t <= end]

    def _windowed_stats(self, history: TimeSeriesBuffer):
        """(count, mean, stdev) for a live-sensor history over the just-
        completed measurement's window, computed from a single snapshot so
        all three are consistent with each other. This is what
        _save_this_test() actually records as the Lux/Irradiance value --
        NOT a live reading taken at the moment Save is clicked, which used
        to drift from what the sensor actually saw during the sweep itself
        depending on how long the operator took to get there (e.g. through
        the mandatory temp prompt) (2026-09 feedback)."""
        windowed = self._windowed_values(history)
        if not windowed:
            return 0, None, None
        mean = float(np.mean(windowed))
        stdev = float(np.std(windowed)) if len(windowed) >= 2 else None
        return len(windowed), mean, stdev

    @staticmethod
    def _three_sigma_pct(stdev, mean):
        """What percentage of the average 3*stddev is -- a quick read on
        how noisy a sensor was during the measurement relative to its own
        level (e.g. 3 sigma at 30% of the mean is a much noisier reading
        than 3 sigma at 2% of the mean, even with the same raw stdev)."""
        if stdev is None or mean is None or mean == 0:
            return None
        return (3.0 * stdev / mean) * 100.0

    def _save_this_test(self):
        if self._last_metrics is None:
            return

        # Lux/irradiance are recorded from what the sensors actually saw
        # DURING the measurement's own window (start of Isc to end of the
        # sweep), not a live reading taken at the moment Save is clicked --
        # the latter used to drift depending on how long the operator took
        # to get here (e.g. through the mandatory temp prompt), so the
        # saved value didn't necessarily reflect the sweep at all (2026-09
        # feedback).
        irr_count, irr_mean, irr_stdev = self._windowed_stats(self.irr_store.history)
        if irr_count < MIN_IRR_SAMPLES:
            window = self._last_measurement_window
            duration_s = (window[1] - window[0]) if window else 0.0
            resp = QMessageBox.question(
                self,
                "Low sample count",
                f"Only {irr_count} irradiance sample(s) were captured during the "
                f"{duration_s:.1f}s measurement itself (minimum {MIN_IRR_SAMPLES}). "
                "Save this row anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return

        _lux_count, lux_mean, lux_stdev = self._windowed_stats(self.lux_history)
        lux_3sigma_pct = self._three_sigma_pct(lux_stdev, lux_mean)
        irr_3sigma_pct = self._three_sigma_pct(irr_stdev, irr_mean)

        snapshot = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "lux": lux_mean if lux_mean is not None else 0.0,
            "irradiance": irr_mean if irr_mean is not None else 0.0,
            "samples": irr_count,
        }

        try:
            summary_csv = self.settings["summary_csv"]
            if self._next_run_index <= 1:
                self._next_run_index = read_max_run_index(summary_csv) + 1
            run_index = self._next_run_index
        except Exception:
            run_index = len(self.session_rows) + 1

        row = make_row(
            session_id="gui-session",
            run_index=run_index,
            status="done",
            panel_name=self.panel_name_edit.text().strip(),
            light_meter=self.light_meter_edit.text().strip(),
            gui_irradiance_input=self.si_edit.text().strip(),
            notes=self.notes_edit.text().strip(),
            metrics=self._last_metrics,
            snapshot=snapshot,
            panel_temp_c=self.panel_temp_edit.text().strip(),
            lux_stdev=lux_stdev if lux_stdev is not None else "",
            irradiance_stdev=irr_stdev if irr_stdev is not None else "",
            lux_3sigma_pct=lux_3sigma_pct if lux_3sigma_pct is not None else "",
            irradiance_3sigma_pct=irr_3sigma_pct if irr_3sigma_pct is not None else "",
        )

        try:
            append_summary_csv(self.settings["summary_csv"], row)
        except Exception as e:
            # Leave _last_metrics/session_rows/btn_save untouched -- i.e.
            # everything exactly as it was before this click -- so the
            # operator can fix the problem (e.g. close the file if it's
            # open in Excel, free up disk space) and click Save This Test
            # again. Previously this branch still cleared _last_metrics and
            # disabled Save, silently forfeiting the row with no retry path
            # (2026-09 adversarial review finding #4).
            self.set_status(
                f"Save failed, nothing was written: {e}. Fix the problem, then click "
                "Save This Test again.",
                "error",
            )
            return

        self.set_status(f"Saved run {run_index} to {self.settings['summary_csv']}", "success")

        self._next_run_index = run_index + 1
        copy_to_clipboard(build_clipboard_row(row))

        self.session_rows.append(row)
        self._append_table_row(row)
        self._refresh_analysis_tab()

        self._last_metrics = None
        self.btn_save.setEnabled(False)
        self._clear_pending_measurement_snapshot()

    def _append_table_row(self, row: dict):
        self.table.blockSignals(True)
        r = self.table.rowCount()
        self.table.insertRow(r)

        def _item(text, editable):
            item = QTableWidgetItem(str(text))
            if not editable:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            return item

        values = [
            row.get("panel_name", ""),
            row.get("timestamp", ""),
            f"{row.get('Wp', ''):.4g}" if row.get("Wp") not in ("", None) else "",
            f"{row.get('Voc', ''):.4g}" if row.get("Voc") not in ("", None) else "",
            f"{row.get('Isc', ''):.4g}" if row.get("Isc") not in ("", None) else "",
            f"{row.get('Vp', ''):.4g}" if row.get("Vp") not in ("", None) else "",
            f"{row.get('Ip', ''):.4g}" if row.get("Ip") not in ("", None) else "",
            row.get("light_meter", ""),
            row.get("irradiance_gui_input", ""),
            row.get("panel_temp_c", ""),
            f"{row.get('lux_measured', ''):.3f}" if row.get("lux_measured") not in ("", None) else "",
            f"{row.get('lux_stdev', ''):.3f}" if row.get("lux_stdev") not in ("", None) else "",
            f"{row.get('lux_3sigma_pct', ''):.1f}" if row.get("lux_3sigma_pct") not in ("", None) else "",
            f"{row.get('irradiance_measured_live', ''):.6f}" if row.get("irradiance_measured_live") not in ("", None) else "",
            f"{row.get('irradiance_stdev', ''):.6f}" if row.get("irradiance_stdev") not in ("", None) else "",
            f"{row.get('irradiance_3sigma_pct', ''):.1f}" if row.get("irradiance_3sigma_pct") not in ("", None) else "",
            row.get("notes", ""),
        ]
        for col, value in enumerate(values):
            self.table.setItem(r, col, _item(value, col in _EDITABLE_COLS))

        self.table.blockSignals(False)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        col = item.column()
        row_idx = item.row()
        field = _FIELD_MAP.get(col)
        if field is None or row_idx >= len(self.session_rows):
            return

        old_value = self.session_rows[row_idx].get(field, "")
        new_value = item.text()
        if new_value == old_value:
            return
        self.session_rows[row_idx][field] = new_value

        # Every row shown in this table is already a saved row (rows only
        # ever get here via _save_this_test() or reloading a working
        # folder's summary CSV) -- so any edit here is, by definition, an
        # edit to an already-persisted measurement. Previously this only
        # updated the in-memory table; the summary CSV on disk (and hence
        # the Analysis tab, which always re-reads it from disk) never saw
        # the edit at all.
        if field == "panel_name":
            old_name, new_name = old_value, new_value
            self.session_rows[row_idx]["panel_type"] = derive_panel_type(new_name)
            self._rename_panel_in_saved_files(row_idx, old_name, new_name)

        self._rewrite_summary_csv()

        if field == "panel_name":
            self._refresh_analysis_tab()

    def _rewrite_summary_csv(self) -> None:
        """Rewrite the whole working folder's summary CSV from
        self.session_rows (the complete, current set of rows -- it mirrors
        the table exactly). append_summary_csv() only ever adds new rows,
        so an edit to an existing row needs this instead."""
        try:
            rewrite_summary_csv(self.settings["summary_csv"], self.session_rows)
        except Exception as e:
            _log("error", "summary_csv_rewrite_failed", error=str(e))
            self.set_status(f"Edit kept in the table, but rewriting the summary CSV failed: {e}", "warning")

    def _rename_panel_in_saved_files(self, row_idx: int, old_name: str, new_name: str) -> None:
        """Renaming a Panel Name in the table should also rename that
        measurement's own IV-curve PNG/CSV files on disk -- they bake the
        panel name into their filename at render time (see render_iv_curve
        in IV_curve_CURRENT_V3.py), so otherwise the files and the table
        would silently disagree about which panel they're for. Uses the
        exact paths recorded on the row at render time (png_path/csv_path)
        rather than reconstructing a filename, since the row's own
        timestamp (set later, after the mandatory temp prompt) doesn't
        necessarily match the one baked into the filename."""
        row = self.session_rows[row_idx]
        renamed, missing, failed = [], [], []
        for path_key in ("png_path", "csv_path"):
            old_path_str = row.get(path_key, "")
            if not old_path_str:
                continue
            old_path = Path(old_path_str)
            if not old_path.exists():
                missing.append(old_path.name)
                continue

            new_name_on_disk = old_path.name.replace(f"Panel {old_name} ", f"Panel {new_name} ", 1)
            if new_name_on_disk == old_path.name:
                # The old name wasn't found verbatim in the filename --
                # don't guess at a rename, just leave the file alone.
                continue

            new_path = old_path.with_name(new_name_on_disk)
            try:
                old_path.rename(new_path)
            except Exception as e:
                _log("error", "rename_saved_file_failed", path=old_path_str, error=str(e))
                failed.append(old_path.name)
                continue
            row[path_key] = str(new_path)
            renamed.append(new_path.name)

        if failed:
            self.set_status(
                f"Renamed \"{old_name}\" to \"{new_name}\" in the table, but couldn't rename "
                f"the saved file(s) on disk: {', '.join(failed)}",
                "warning",
            )
        elif missing:
            self.set_status(
                f"Renamed \"{old_name}\" to \"{new_name}\" in the table, but the saved file(s) "
                f"recorded for this row are missing on disk: {', '.join(missing)}",
                "warning",
            )
        elif renamed:
            self.set_status(f"Renamed \"{old_name}\" to \"{new_name}\" — updated {len(renamed)} saved file(s) too.", "success")

    def _copy_table_selection(self):
        selection = self.table.selectedItems()
        if not selection:
            return
        rows = sorted({item.row() for item in selection})
        cols = sorted({item.column() for item in selection})
        lines = []
        for r in rows:
            cells = []
            for c in cols:
                item = self.table.item(r, c)
                cells.append(item.text() if item else "")
            lines.append("\t".join(cells))
        copy_to_clipboard("\n".join(lines))

    def _save_all_and_finish(self):
        if not self.session_rows:
            self.set_status("No measurements saved yet in this session", "warning")
            return

        block = "".join(build_clipboard_row(row) for row in self.session_rows)
        copy_to_clipboard(block)

        try:
            open_in_explorer(str(self.working_dir))
        except Exception:
            pass

        self.set_status(
            f"Session finished — {len(self.session_rows)} measurement(s) saved, "
            "full session copied to clipboard, output folder opened.",
            "success",
        )

    def _start_over(self):
        resp = QMessageBox.question(
            self,
            "Start Over",
            "Clear the session table? Rows already written to the summary CSV are not deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return

        self.session_rows.clear()
        self.table.setRowCount(0)
        self._last_metrics = None
        self._last_measurement_window = None
        self._measurement_markers.clear()
        self._redraw_timeseries()
        # Note: the summary CSV itself isn't touched by Start Over (see the
        # confirmation text above), so this is a no-op visually today -- but
        # keeps the Analysis tab consistent with the table if that changes.
        self._refresh_analysis_tab()
        self.results_lbl.setText("—")
        self.btn_save.setEnabled(False)
        self.set_status("Session cleared", "info")


class _SafeApplication(QApplication):
    """QApplication.notify() is where Qt's C++ event loop calls back into a
    Python slot (button clicks, table-item-changed, filter changes, hover,
    ...) -- letting a Python exception propagate out through it is what
    aborts PyQt6 outright (this project hit exactly that crash earlier, see
    the AsyncWorker comment above). _launch_task() already guards the
    handful of slots that schedule an async task this way; overriding
    notify() extends the same net to every other plain slot too (2026-09
    adversarial review finding #8) -- a last resort, not a substitute for
    each slot handling its own expected failures."""

    def notify(self, receiver, event) -> bool:
        try:
            return super().notify(receiver, event)
        except Exception as e:
            _log("error", "unhandled_slot_exception", error=str(e))
            try:
                for widget in self.topLevelWidgets():
                    if isinstance(widget, LowLightApp):
                        widget.set_status(f"Internal error (caught): {e}", "error")
                        break
            except Exception:
                pass
            return False


def main():
    app = _SafeApplication(sys.argv)
    window = LowLightApp()
    window.showMaximized()

    def _handle_exception(loop, context):
        # Last-resort net for anything that escapes a task's own try/except
        # (each async method already catches its own exceptions and reports
        # them via set_status(), same as before -- this just guarantees an
        # uncaught one can never crash the async worker thread).
        message = context.get("exception") or context.get("message")
        _log("error", "unhandled_async_exception", detail=str(message))
        try:
            window.set_status(f"Internal error: {message}", "error")
        except Exception:
            pass

    window._async_worker.loop.call_soon_threadsafe(
        window._async_worker.loop.set_exception_handler, _handle_exception
    )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
