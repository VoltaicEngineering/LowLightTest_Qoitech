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
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports

from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QKeySequence, QShortcut
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

import lightbox_mode as lbm
import norm_analysis
import sensors
import zero_offset
from triplett_lt68_probe_v3 import LT68
from listen import RollingIrradiance, serial_reader
from sensors import (
    CACHE_DIR,
    LUX_TIMEOUT_S,
    LUX_SETTLE_S,
    LUX_POLL_INTERVAL_S,
    IRR_BAUD_DEFAULT,
    IRR_WINDOW_S,
    MIN_IRR_SAMPLES,
    LIVE_STALE_S,
    TIMESERIES_HISTORY_S,
    TIMESERIES_REDRAW_MS,
    detect_sensor_ports,
    load_cached_port,
    save_cached_port,
    LiveValue,
    TimeSeriesBuffer,
    HistoryIrradiance,
    lux_poll_loop,
)
from ui_style import _STYLESHEET, _STAGE_STYLE, apply_light_theme, open_in_explorer
from IV_curve_CURRENT_V3 import (
    append_summary_csv,
    build_clipboard_row,
    connect_otii_session,
    copy_to_clipboard,
    delete_recording,
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
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "report" / "LowLightTesting_summary.csv"

# Stable, deterministic color assignment (by sorted group label) for the
# Analysis tab's 4 graphs, so a given panel keeps the same color
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
_NOTES_COL = 16
# Lightbox mode only (hidden otherwise), appended after Notes so the indices
# above never move: (header, row key, format spec or None for plain text).
_LIGHTBOX_COLUMNS = [
    ("Setpoint (W/m²)", "lightbox_setpoint_wm2", "g"),
    ("VAC Set", "vac_set", ".1f"),
    ("Irr Node", "irr_node", None),
    ("Irr Exp (W/m²)", "irr_expected", ".4g"),
    ("Irr Dev (%)", "irr_deviation_pct", "+.1f"),
    ("Lux Node", "lux_node", None),
    ("Lux Exp (lx)", "lux_expected", ".4g"),
    ("Lux Dev (%)", "lux_deviation_pct", "+.1f"),
]
_COLUMNS = _COLUMNS + [c[0] for c in _LIGHTBOX_COLUMNS]


def _log(level: str, event: str, **fields) -> None:
    sensors._log(level, event, **fields)


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
    # Delete each recording from the Otii project once its data is saved.
    # Otii slows down as a project fills with recordings.
    "otii_delete_recordings": True,
    "summary_csv": str(DEFAULT_SUMMARY_CSV),
    "working_dir": "",  # "" = use the process's current working directory
    "panel_config_path": str(DEFAULT_PANEL_CONFIG_PATH),
    "reference_tolerance_pct": norm_analysis.DEFAULT_TOLERANCE * 100.0,
    # Lightbox mode (--lightbox)
    "lightbox_campaign_path": "",  # last calibration folder chosen; none built in
    "lightbox_deviation_warn_pct": 5.0,
    "lightbox_trim_tolerance_pct": 2.0,
    "lightbox_irr_node": lbm.DEFAULT_IRR_NODE,
    "lightbox_lux_node": lbm.DEFAULT_LUX_NODE,
}

# Settings tab: description and validation per key. kind: "path" / "text" /
# "float" (min, strict) / "node". read_only keys are shown but set elsewhere.
_SETTINGS_META = {
    "otii_exe": ("path", "Otii 3 executable, used to restart Otii if it stops responding."),
    "otii_restart_wait_seconds": ("float", "Seconds to wait after restarting Otii before reconnecting.", 0.0, False),
    "iv_timeout_seconds": ("float", "Maximum IV sweep time before it is aborted (same as Sweep Timeout on the Measure tab).", 0.0, True),
    "otii_call_timeout_seconds": ("float", "Timeout for a single Otii call before the connection counts as lost.", 0.0, True),
    "otii_delete_recordings": ("bool", "Delete each recording from the Otii project once its IV data is saved (keeps later sweeps fast). false keeps them in Otii."),
    "summary_csv": ("path", "Summary CSV. Always <working folder>/LowLightTesting_summary.csv; change the working folder instead.", None, None, True),
    "working_dir": ("path", "Working folder for IV files and the summary CSV (the session there is reloaded)."),
    "panel_config_path": ("path", "Panel config CSV (panel name, cells, width/height)."),
    "reference_tolerance_pct": ("float", "Analysis tab: measurements within this % of 1000 W/m² form each panel's reference.", 0.0, True),
    "lightbox_campaign_path": ("path", "Lightbox: calibration campaign folder (Choose… on the Measure tab sets it too)."),
    "lightbox_deviation_warn_pct": ("float", "Lightbox: warn at save when a sensor is further than this % from the calibration.", 0.0, False),
    "lightbox_trim_tolerance_pct": ("float", "Lightbox: the trim dialog shows green within this % of the target.", 0.0, True),
    "lightbox_irr_node": ("node", "Lightbox: default irradiance sensor grid node."),
    "lightbox_lux_node": ("node", "Lightbox: default lux sensor grid node."),
}


def parse_setting_value(key: str, text: str, default):
    """Parse one Settings-tab cell into the value stored in settings.json.
    Raises ValueError with a readable reason."""
    meta = _SETTINGS_META.get(key)
    text = text.strip()
    kind = meta[0] if meta else ("float" if isinstance(default, (int, float)) and not isinstance(default, bool) else "json")
    if kind == "float":
        try:
            value = float(text)
        except ValueError:
            raise ValueError("must be a number") from None
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("must be a finite number")
        if meta and meta[2] is not None:
            minimum, strict = meta[2], meta[3]
            if (strict and value <= minimum) or (not strict and value < minimum):
                raise ValueError(f"must be {'>' if strict else '>='} {minimum:g}")
        return value
    if kind == "bool":
        low = text.lower()
        if low in ("true", "yes", "1", "on"):
            return True
        if low in ("false", "no", "0", "off"):
            return False
        raise ValueError("must be true or false")
    if kind == "node":
        try:
            import calibration_model as _cm
            _cm.parse_node_label(text)
        except ValueError:
            raise ValueError("must be a grid node like B6") from None
        return text.upper()
    if kind == "json":
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text


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
        self.lb_dev_edit = QLineEdit(str(settings.get("lightbox_deviation_warn_pct", 5.0)))
        self.lb_trim_edit = QLineEdit(str(settings.get("lightbox_trim_tolerance_pct", 2.0)))

        form.addRow("Otii 3 executable path:", self.otii_exe_edit)
        form.addRow("Otii restart wait (s):", self.restart_wait_edit)
        form.addRow("IV sweep timeout (s):", self.iv_timeout_edit)
        form.addRow("Otii call timeout (s):", self.call_timeout_edit)
        form.addRow("Summary CSV path:", self.summary_csv_edit)
        form.addRow("Panel config file:", self.panel_config_edit)
        form.addRow("1000 W/m² reference tolerance (%):", self.tolerance_edit)
        form.addRow("Lightbox: warn when a sensor deviates by more than (%):", self.lb_dev_edit)
        form.addRow("Lightbox: trim tolerance, shown green within (%):", self.lb_trim_edit)
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
            "lightbox_deviation_warn_pct": abs(_float(self.lb_dev_edit, 5.0)),
            "lightbox_trim_tolerance_pct": abs(_float(self.lb_trim_edit, 2.0)),
        }


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class LowLightApp(QMainWindow):
    MAX_AUTO_RESTARTS = 2
    AUTO_RESTART_WINDOW_S = 5 * 60.0
    _ui_call = pyqtSignal(object)  # marshals a zero-arg callable to the main thread

    def __init__(self, lightbox: bool = False):
        super().__init__()
        self.settings = load_settings()
        # Lightbox mode: calibrated indoor lightbox, one setpoint per round
        # (see lightbox_mode.py). Everything it adds is hidden otherwise.
        self.lightbox_mode = lightbox
        self.lb_calibration: lbm.Calibration | None = None
        self.lb_round: lbm.Round | None = None
        self._last_lightbox_ctx: dict | None = None
        # Irradiance zero offset captured on the Irr Zero Offset tab. Applies
        # for the rest of this session only (live readout, plot, logged
        # values); not saved, so each session starts uncorrected.
        self.irr_zero_offset: float | None = None
        self.irr_zero_offset_info = ""

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

        if self.lightbox_mode:
            saved = self.settings.get("lightbox_campaign_path") or ""
            if saved:
                self._lightbox_load_calibration(saved, quiet=True)
            self._lightbox_refresh_info()

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
        self.setWindowTitle(
            "Low Light IV Tester — Lightbox — Voltaic Systems" if self.lightbox_mode
            else "Low Light IV Tester — Voltaic Systems"
        )
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

        title = QLabel("LOW LIGHT IV TESTER — LIGHTBOX" if self.lightbox_mode else "LOW LIGHT IV TESTER")
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
        # Same setting as Settings > "IV sweep timeout (s)", surfaced here so
        # it can be changed between runs without opening the dialog.
        self.sweep_timeout_edit = QLineEdit(f"{self.settings['iv_timeout_seconds']:g}")
        self.sweep_timeout_edit.setToolTip(
            "Maximum time the IV sweep may run before it is aborted. Takes effect on the next Run Test."
        )
        self.sweep_timeout_edit.editingFinished.connect(self._guard(self._commit_sweep_timeout))
        panel_form.addRow("Sweep Timeout (s):", self.sweep_timeout_edit)
        clay.addLayout(panel_form)
        if self.lightbox_mode:
            # The round's setpoint is the solar intensity; it's written into
            # si_edit so file names, plot titles and the saved row use it.
            panel_form.setRowVisible(self.si_edit, False)
            self.si_edit.setText("")
            clay.addWidget(self._build_lightbox_group())

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
        for col in range(1, len(_COLUMNS)):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        if self.lightbox_mode:
            hdr.setSectionResizeMode(_NOTES_COL, QHeaderView.ResizeMode.Interactive)
            self.table.setColumnWidth(_NOTES_COL, 160)
        else:
            hdr.setSectionResizeMode(_NOTES_COL, QHeaderView.ResizeMode.Stretch)
            for col in range(_NOTES_COL + 1, len(_COLUMNS)):
                self.table.setColumnHidden(col, True)
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
        self._tab_widget.addTab(self._build_zero_offset_tab(), "Irr Zero Offset")
        self._settings_tab_index = self._tab_widget.addTab(self._build_settings_tab(), "Settings")
        # Other controls (Sweep Timeout box, working folder, calibration
        # Choose…, the Settings dialog) also change settings, so the tab is
        # re-read whenever it's opened unless it holds unsaved edits.
        self._tab_widget.currentChanged.connect(self._guard1(self._on_tab_changed))

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

        # Filters: narrow which panels are plotted. Applied at
        # display time only (see _refresh_analysis_tab) -- they never change
        # which measurements feed into a panel's 1000 W/m^2 reference, only
        # what's drawn, so hiding a panel can't quietly shift another one's
        # normalization.
        filter_row = QHBoxLayout()
        self._filter_panel_list = self._make_filter_list()
        self._filter_celltype_list = self._make_filter_list()
        self._filter_series_list = self._make_filter_list()
        self._filter_parallel_list = self._make_filter_list()
        for label, widget in (
            ("Panel Name", self._filter_panel_list),
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
            "One row per panel, keyed by the exact Panel Name entered on the Measure tab "
            "(case-insensitive). Each panel is normalized against its own 1000 W/m² "
            "measurements on the Analysis tab. Width and Height are only needed for the "
            "lightbox: width runs along columns A–K and should be the longer side. Edits here "
            "save to the panel config file and refresh the Analysis tab automatically."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#555; font-size:9pt;")
        layout.addWidget(intro)

        self.panel_config_status_lbl = QLabel("")
        self.panel_config_status_lbl.setWordWrap(True)
        self.panel_config_status_lbl.setStyleSheet("color:#444; font-size:9pt;")
        layout.addWidget(self.panel_config_status_lbl)

        self.panel_config_table = QTableWidget()
        self.panel_config_table.setColumnCount(6)
        self.panel_config_table.setHorizontalHeaderLabels(
            ["Panel Name", "Cell Type", "Cells Series", "Cells Parallel", "Width (mm)", "Height (mm)"]
        )
        hdr = self.panel_config_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        self.panel_config_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.panel_config_table.itemChanged.connect(self._guard(self._on_panel_config_table_changed))
        layout.addWidget(self.panel_config_table, 1)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Row")
        add_btn.clicked.connect(self._guard(self._add_panel_config_row))
        add_measured_btn = QPushButton("Add Measured Panels")
        add_measured_btn.setToolTip(
            "Add a row for every Panel Name in the current working folder's results "
            "that has no config yet -- fill in Cells Series/Parallel to include it"
        )
        add_measured_btn.clicked.connect(self._guard(self._add_measured_panel_rows))
        delete_btn = QPushButton("Delete Selected Row(s)")
        delete_btn.setProperty("danger", True)
        delete_btn.clicked.connect(self._guard(self._delete_panel_config_rows))
        reload_btn = QPushButton("Reload From File")
        reload_btn.setToolTip("Discard unsaved table edits and re-read the file from disk")
        reload_btn.clicked.connect(self._guard(self._reload_panel_config))
        btn_row.addWidget(add_btn)
        btn_row.addWidget(add_measured_btn)
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

        offset = self._active_irr_offset()[0]
        self.irr_ts_ax.set_title(
            "Irradiance (live)" if offset is None else f"Irradiance (live, {offset:.4f} offset subtracted)", fontsize=9
        )
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
        for name, entry in sorted(self.panel_config.items(), key=lambda kv: norm_analysis.normalize_panel_name(kv[0])):
            r = self.panel_config_table.rowCount()
            self.panel_config_table.insertRow(r)
            self.panel_config_table.setItem(r, 0, QTableWidgetItem(name))
            self.panel_config_table.setItem(r, 1, QTableWidgetItem(entry["cell_type"]))
            self.panel_config_table.setItem(r, 2, QTableWidgetItem(str(entry["cells_series"])))
            self.panel_config_table.setItem(r, 3, QTableWidgetItem(str(entry["cells_parallel"])))
            for c, key in ((4, "width_mm"), (5, "height_mm")):
                value = entry.get(key)
                self.panel_config_table.setItem(r, c, QTableWidgetItem("" if value is None else f"{value:g}"))
        self.panel_config_table.blockSignals(False)

    def _add_panel_config_row(self) -> None:
        self.panel_config_table.blockSignals(True)
        r = self.panel_config_table.rowCount()
        self.panel_config_table.insertRow(r)
        for c, default in enumerate(("", "", "1", "1", "", "")):
            self.panel_config_table.setItem(r, c, QTableWidgetItem(default))
        self.panel_config_table.blockSignals(False)
        self.panel_config_table.editItem(self.panel_config_table.item(r, 0))
        # Left blank (no panel name) until the operator actually edits a
        # cell -- _on_panel_config_table_changed() skips name-less rows
        # rather than saving/plotting an incomplete one.

    def _read_summary_rows(self) -> list:
        summary_path = self.working_dir / SESSION_CSV_NAME
        if not summary_path.exists():
            return []
        with summary_path.open("r", newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def _add_measured_panel_rows(self) -> None:
        """Append a row for each Panel Name in the working folder's results
        that isn't in the table yet. Series/Parallel are left blank, so the
        row isn't saved until the operator fills them in."""
        try:
            rows = self._read_summary_rows()
        except Exception as e:
            self.panel_config_status_lbl.setText(f"Could not read measured panels: {e}")
            return
        present = set()
        for r in range(self.panel_config_table.rowCount()):
            item = self.panel_config_table.item(r, 0)
            if item and item.text().strip():
                present.add(norm_analysis.normalize_panel_name(item.text()))
        missing = {}
        for row in rows:
            name = (row.get("panel_name") or "").strip()
            key = norm_analysis.normalize_panel_name(name)
            if key and key not in present and key not in missing:
                missing[key] = name
        if not missing:
            self.panel_config_status_lbl.setText("Every measured panel already has a row.")
            return
        self.panel_config_table.blockSignals(True)
        for key in sorted(missing):
            r = self.panel_config_table.rowCount()
            self.panel_config_table.insertRow(r)
            for c, value in enumerate((missing[key], "", "", "", "", "")):
                self.panel_config_table.setItem(r, c, QTableWidgetItem(value))
        self.panel_config_table.blockSignals(False)
        self.panel_config_status_lbl.setText(
            f"Added {len(missing)} measured panel(s). Fill in Cells Series/Parallel to save and analyse them."
        )

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
        incomplete_rows = 0
        duplicate_rows = 0
        invalid_dims = 0
        seen = set()

        def _text(r, c):
            item = self.panel_config_table.item(r, c)
            return item.text().strip() if item else ""

        for r in range(self.panel_config_table.rowCount()):
            name = _text(r, 0)
            if not name:
                continue  # incomplete/new row -- not an error, just not ready yet
            if not _text(r, 2) or not _text(r, 3):
                incomplete_rows += 1  # e.g. from "Add Measured Panels", not filled in yet
                continue
            key = norm_analysis.normalize_panel_name(name)
            if key in seen:
                duplicate_rows += 1
                continue
            try:
                cells_series = int(float(_text(r, 2)))
                cells_parallel = int(float(_text(r, 3)))
            except ValueError:
                invalid_rows += 1
                continue
            try:
                width_mm = norm_analysis.parse_dimension_mm(_text(r, 4))
                height_mm = norm_analysis.parse_dimension_mm(_text(r, 5))
            except ValueError:
                invalid_dims += 1
                continue
            seen.add(key)
            config[name] = {
                "cell_type": _text(r, 1),
                "cells_series": cells_series,
                "cells_parallel": cells_parallel,
                "width_mm": width_mm,
                "height_mm": height_mm,
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
        if incomplete_rows:
            status += f" {incomplete_rows} row(s) not saved yet (Cells Series/Parallel blank)."
        if duplicate_rows:
            status += f" {duplicate_rows} duplicate Panel Name row(s) ignored (first one kept)."
        if invalid_dims:
            status += f" {invalid_dims} row(s) skipped (Width/Height must be blank or a number > 0)."
        self.panel_config_status_lbl.setText(status)
        self.panel_config_lbl.setText(f"Panel config: {path} ({len(self.panel_config)} entries)")

        self._populate_analysis_filters()
        self._refresh_analysis_tab()

    def _populate_analysis_filters(self) -> None:
        """(Re)populate the 4 filter lists from the currently-loaded panel
        config, preserving each list's current selection where the value
        still exists."""
        panel_names = sorted(self.panel_config.keys(), key=norm_analysis.normalize_panel_name)
        cell_types = sorted({e["cell_type"] for e in self.panel_config.values() if e["cell_type"]})
        series_vals = sorted({str(e["cells_series"]) for e in self.panel_config.values()}, key=lambda s: float(s))
        parallel_vals = sorted({str(e["cells_parallel"]) for e in self.panel_config.values()}, key=lambda s: float(s))

        for widget, values in (
            (self._filter_panel_list, panel_names),
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
            self._filter_panel_list, self._filter_celltype_list,
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
        try:
            rows = self._read_summary_rows()
        except Exception as e:
            self.analysis_status_lbl.setText(f"Could not read {self.working_dir / SESSION_CSV_NAME}: {e}")
            return

        tolerance = self.settings.get("reference_tolerance_pct", 3.0) / 100.0
        result = norm_analysis.compute_norm_dataset(rows, self.panel_config, tolerance=tolerance)
        groups = result["groups"]

        selected_panels = {
            norm_analysis.normalize_panel_name(item.text()) for item in self._filter_panel_list.selectedItems()
        }
        selected_cell_types = {item.text() for item in self._filter_celltype_list.selectedItems()}
        selected_series = {item.text() for item in self._filter_series_list.selectedItems()}
        selected_parallel = {item.text() for item in self._filter_parallel_list.selectedItems()}

        def group_visible(group_key, group_data) -> bool:
            cell_type = group_data["cell_type"]
            series, parallel = group_data["cells_series"], group_data["cells_parallel"]
            if selected_panels and group_key not in selected_panels:
                return False
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
            if not group_visible(group_key, group_data):
                continue
            points = group_data["points"]
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

        if plotted_groups:
            # SI as a % of 1000 W/m²: where a value exactly proportional to
            # irradiance would sit. axline leaves the autoscaled limits alone.
            for ax in axes:
                ax.axline(
                    (0.0, 0.0), slope=100.0 / norm_analysis.REFERENCE_IRRADIANCE,
                    color="#888888", linestyle=":", linewidth=1.2, zorder=1,
                    label=f"SI as % of {norm_analysis.REFERENCE_IRRADIANCE:g} W/m²",
                )

        self._style_analysis_axes()
        if plotted_groups:
            self._norm_wp_ax.legend(fontsize=6, loc="best")
        self.analysis_canvas.figure.tight_layout()
        self.analysis_canvas.draw_idle()

        filters_active = bool(selected_panels or selected_cell_types or selected_series or selected_parallel)
        filter_note = " (filtered)" if filters_active else ""
        self.analysis_status_lbl.setText(
            f"{plotted_groups} panel(s), {plotted_points} point(s) plotted{filter_note} — "
            f"{result['skipped_no_config']} skipped (no panel config match), "
            f"{result['skipped_no_reference']} skipped (no 1000 W/m² reference yet), "
            f"{result['skipped_bad_data']} skipped (missing/bad data)"
        )

    def _active_irr_offset(self):
        """(offset, label) subtracted from irradiance shown and logged now:
        the lightbox round's dark offset during a round, otherwise this
        session's zero offset, otherwise (None, "")."""
        if self.lightbox_mode and self.lb_round is not None:
            return self.lb_round.dark_offset, f"round dark offset, {self.lb_round.dark_offset_source}"
        if self.irr_zero_offset is not None:
            return self.irr_zero_offset, "zero offset"
        return None, ""

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
            offset = self._active_irr_offset()[0] or 0.0
            self.irr_ts_ax.plot([t - now for t in irr_times], [v - offset for v in irr_values],
                                color="#c53030", linewidth=1.2)

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

    # ------------------------------------------------------------------
    # Settings tab: edit cache/settings.json in place
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Irr Zero Offset tab: capture the irradiance sensor's reading at zero
    # irradiance (sensor covered) and keep it as the zero offset.
    # ------------------------------------------------------------------

    def _build_zero_offset_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        intro = QLabel(
            "Cover the irradiance sensor completely (no light at all), wait for the reading to settle, "
            "then Start Capture. Stop Capture once the running average has levelled off: the average "
            "of every sample captured becomes the zero offset (the reading at 0 W/m²). For the rest of "
            "this session it is subtracted from the live irradiance readout, the irradiance plot and "
            "every saved test (irradiance_measured_live; the raw mean goes in irradiance_raw and the "
            "offset in irr_dark_offset). It is not kept after the app closes. In lightbox mode it "
            "also becomes the current round's dark offset. The plot below always shows raw readings."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#555; font-size:9pt;")
        layout.addWidget(intro)

        row = QHBoxLayout()
        self.zo_start_btn = QPushButton("Start Capture")
        self.zo_start_btn.setProperty("primary", True)
        self.zo_start_btn.clicked.connect(self._guard(self._zero_offset_start))
        self.zo_stop_btn = QPushButton("Stop Capture")
        self.zo_stop_btn.setEnabled(False)
        self.zo_stop_btn.clicked.connect(self._guard(self._zero_offset_stop))
        self.zo_live_lbl = QLabel("Not capturing.")
        self.zo_live_lbl.setStyleSheet("font-size:11pt; font-weight:bold;")
        self.zo_clear_btn = QPushButton("Clear Offset")
        self.zo_clear_btn.setToolTip("Stop subtracting a zero offset for the rest of this session")
        self.zo_clear_btn.setEnabled(False)
        self.zo_clear_btn.clicked.connect(self._guard(self._zero_offset_clear))
        row.addWidget(self.zo_start_btn)
        row.addWidget(self.zo_stop_btn)
        row.addWidget(self.zo_clear_btn)
        row.addSpacing(12)
        row.addWidget(self.zo_live_lbl, 1)
        layout.addLayout(row)

        self.zo_current_lbl = QLabel("")
        self.zo_current_lbl.setWordWrap(True)
        self.zo_current_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.zo_current_lbl.setStyleSheet("font-size:10pt;")
        layout.addWidget(self.zo_current_lbl)

        fig = Figure(dpi=100, facecolor="#f0f2f5")
        self.zo_canvas = FigureCanvas(fig)
        self.zo_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.zo_ax = fig.add_subplot(111)
        layout.addWidget(self.zo_canvas, 1)

        self.zo_capture = zero_offset.ZeroOffsetCapture(self.irr_store.history)
        self._zo_timer = QTimer(self)
        self._zo_timer.setInterval(250)
        self._zo_timer.timeout.connect(self._guard(self._zero_offset_tick))
        self._zero_offset_refresh_current()
        self._zero_offset_draw()
        return tab

    def _zero_offset_refresh_current(self) -> None:
        value = self.irr_zero_offset
        if value is None:
            text = "Session zero offset: none. Irradiance is shown and logged uncorrected."
        else:
            text = (f"Session zero offset: <b>{value:.4f} W/m²</b> ({self.irr_zero_offset_info}), "
                    "subtracted from irradiance until the app is closed.")
        if self.lightbox_mode and self.lb_round is not None:
            text += (f"<br>Lightbox round in progress: its dark offset "
                     f"{self.lb_round.dark_offset:.4f} W/m² ({self.lb_round.dark_offset_source}) is what's applied.")
        self.zo_current_lbl.setText(text)
        self.zo_clear_btn.setEnabled(value is not None)

    def _zero_offset_clear(self) -> None:
        self.irr_zero_offset = None
        self.irr_zero_offset_info = ""
        self._zero_offset_refresh_current()
        self._redraw_timeseries()
        self.set_status("Irradiance zero offset cleared: irradiance is uncorrected for the rest of this session", "info")

    def _zero_offset_start(self) -> None:
        self.zo_capture.start()
        self.zo_start_btn.setEnabled(False)
        self.zo_stop_btn.setEnabled(True)
        self.zo_live_lbl.setStyleSheet("font-size:11pt; font-weight:bold;")
        self._zo_timer.start()
        self._zero_offset_tick()

    def _zero_offset_tick(self) -> None:
        cap = self.zo_capture
        cap.poll()
        mean, sd = cap.mean(), cap.stdev()
        parts = [f"{'Capturing' if cap.running else 'Captured'} {cap.elapsed():.0f} s", f"{cap.n} samples"]
        if mean is not None:
            parts.append(f"running average {mean:.4f} W/m²")
        if sd is not None:
            parts.append(f"std dev {sd:.4f}")
        if cap.running and cap.n == 0 and cap.elapsed() > 5:
            parts.append("no samples: is the irradiance sensor connected?")
        self.zo_live_lbl.setText("   ·   ".join(parts))
        self._zero_offset_draw()

    def _zero_offset_stop(self) -> None:
        cap = self.zo_capture
        cap.stop()
        self._zo_timer.stop()
        self.zo_start_btn.setEnabled(True)
        self.zo_stop_btn.setEnabled(False)
        self._zero_offset_tick()
        if cap.n < MIN_IRR_SAMPLES:
            self.zo_live_lbl.setText(
                f"Only {cap.n} sample(s) captured (minimum {MIN_IRR_SAMPLES}); zero offset not changed."
            )
            self.zo_live_lbl.setStyleSheet("font-size:11pt; font-weight:bold; color:#c53030;")
            return
        self.zo_live_lbl.setStyleSheet("font-size:11pt; font-weight:bold; color:#1b8a5a;")
        mean = cap.mean()
        self.irr_zero_offset = mean
        self.irr_zero_offset_info = (f"captured {time.strftime('%H:%M:%S', time.localtime(cap.stopped_at))}, "
                                     f"{cap.n} samples over {cap.elapsed():.0f} s")
        msg = f"Irradiance zero offset {mean:.4f} W/m² ({cap.n} samples) is now subtracted from irradiance"
        if self.lightbox_mode and self.lb_round is not None:
            # A round subtracts its own dark offset; the new capture replaces it.
            self.lb_round.dark_offset = mean
            self.lb_round.dark_offset_source = "zero-offset tab"
            self._lightbox_refresh_info()
            msg += " (and is the current round's dark offset)"
        self._zero_offset_refresh_current()
        self._redraw_timeseries()
        self.set_status(msg, "success")

    def _zero_offset_draw(self) -> None:
        cap, ax = self.zo_capture, self.zo_ax
        ax.clear()
        if cap.n:
            t0 = cap.started_at
            xs = [t - t0 for t in cap.times]
            ax.plot(xs, cap.values, color="#1a56db", linewidth=1, marker=".", markersize=3, label="Irradiance")
            ax.plot(xs, cap.running_mean(), color="#c53030", linewidth=2, label="Running average")
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Start Capture with the irradiance sensor covered",
                    ha="center", va="center", transform=ax.transAxes, color="#888")
        ax.set_title("Irradiance zero-offset capture", fontsize=10)
        ax.set_xlabel("Seconds since Start Capture", fontsize=9)
        ax.set_ylabel("W/m²", fontsize=9)
        ax.grid(True, alpha=0.3)
        self.zo_canvas.figure.tight_layout()
        self.zo_canvas.draw_idle()

    def _build_settings_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        intro = QLabel(
            f"Every value in {CACHE_DIR / 'settings.json'}. Edit a Value cell, then Apply && Save. "
            "Invalid entries are highlighted and nothing is saved until they're fixed. "
            "Grey rows are set elsewhere and shown for reference."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#555; font-size:9pt;")
        layout.addWidget(intro)

        self.settings_status_lbl = QLabel("")
        self.settings_status_lbl.setWordWrap(True)
        self.settings_status_lbl.setStyleSheet("color:#444; font-size:9pt;")
        layout.addWidget(self.settings_status_lbl)

        self.settings_table = QTableWidget()
        self.settings_table.setColumnCount(4)
        self.settings_table.setHorizontalHeaderLabels(["Setting", "Value", "Default", "Description"])
        hdr = self.settings_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.settings_table.setColumnWidth(1, 380)
        self.settings_table.setColumnWidth(2, 160)
        self.settings_table.setWordWrap(False)
        self.settings_table.itemChanged.connect(self._guard1(self._on_settings_item_changed))
        layout.addWidget(self.settings_table, 1)

        btn_row = QHBoxLayout()
        apply_btn = QPushButton("Apply && Save")
        apply_btn.setProperty("primary", True)
        apply_btn.clicked.connect(self._guard(self._apply_settings_tab))
        discard_btn = QPushButton("Discard Edits")
        discard_btn.setToolTip("Re-read the current settings, dropping unsaved edits in this table")
        discard_btn.clicked.connect(self._guard(self._populate_settings_tab))
        defaults_btn = QPushButton("Reset Row to Default")
        defaults_btn.setToolTip("Put the default value into the selected row(s); Apply && Save to keep it")
        defaults_btn.clicked.connect(self._guard(self._reset_settings_rows_to_default))
        for b in (apply_btn, discard_btn, defaults_btn):
            btn_row.addWidget(b)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._settings_dirty = False
        self._populate_settings_tab()
        return tab

    @staticmethod
    def _setting_text(value) -> str:
        if isinstance(value, float):
            return f"{value:g}"
        if isinstance(value, str):
            return value
        return json.dumps(value)

    def _settings_keys(self) -> list:
        return list(DEFAULT_SETTINGS) + sorted(k for k in self.settings if k not in DEFAULT_SETTINGS)

    def _populate_settings_tab(self) -> None:
        t = self.settings_table
        t.blockSignals(True)
        t.setRowCount(0)
        for key in self._settings_keys():
            meta = _SETTINGS_META.get(key)
            read_only = bool(meta and len(meta) > 4 and meta[4])
            r = t.rowCount()
            t.insertRow(r)
            items = [
                QTableWidgetItem(key),
                QTableWidgetItem(self._setting_text(self.settings.get(key, DEFAULT_SETTINGS.get(key, "")))),
                QTableWidgetItem(self._setting_text(DEFAULT_SETTINGS[key]) if key in DEFAULT_SETTINGS else "(not a built-in setting)"),
                QTableWidgetItem(meta[1] if meta else ""),
            ]
            for c, item in enumerate(items):
                if c != 1 or read_only:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if read_only:
                    item.setForeground(QBrush(QColor("#888888")))
                item.setToolTip(items[1].text() if c == 1 else items[3].text())
                t.setItem(r, c, item)
        t.blockSignals(False)
        self._settings_dirty = False
        self.settings_status_lbl.setText(f"{len(self._settings_keys())} settings loaded.")
        self.settings_status_lbl.setStyleSheet("color:#444; font-size:9pt;")

    def _on_tab_changed(self, index: int) -> None:
        self._zero_offset_refresh_current()
        if index == self._settings_tab_index and not self._settings_dirty:
            self._populate_settings_tab()

    def _on_settings_item_changed(self, item) -> None:
        if item.column() != 1:
            return
        self._settings_dirty = True
        key = self.settings_table.item(item.row(), 0).text()
        self.settings_table.blockSignals(True)
        try:
            parse_setting_value(key, item.text(), DEFAULT_SETTINGS.get(key))
            item.setBackground(QBrush())
            item.setToolTip(item.text())
        except ValueError as e:
            item.setBackground(QBrush(QColor("#ffe08a")))
            item.setToolTip(f"{key} {e}")
        finally:
            self.settings_table.blockSignals(False)
        self.settings_status_lbl.setText("Unsaved edits: click Apply && Save.")
        self.settings_status_lbl.setStyleSheet("color:#8a5300; font-size:9pt; font-weight:bold;")

    def _reset_settings_rows_to_default(self) -> None:
        rows = sorted({i.row() for i in self.settings_table.selectedItems()})
        if not rows:
            self.settings_status_lbl.setText("Select the row(s) to reset first.")
            return
        for r in rows:
            key = self.settings_table.item(r, 0).text()
            value_item = self.settings_table.item(r, 1)
            if key in DEFAULT_SETTINGS and value_item.flags() & Qt.ItemFlag.ItemIsEditable:
                value_item.setText(self._setting_text(DEFAULT_SETTINGS[key]))

    def _apply_settings_tab(self) -> None:
        """Validate every row, then save and apply what changed. Nothing is
        saved if any row is invalid."""
        new, errors = {}, []
        for r in range(self.settings_table.rowCount()):
            key = self.settings_table.item(r, 0).text()
            item = self.settings_table.item(r, 1)
            if not item.flags() & Qt.ItemFlag.ItemIsEditable:
                continue
            try:
                new[key] = parse_setting_value(key, item.text(), DEFAULT_SETTINGS.get(key))
            except ValueError as e:
                errors.append(f"{key} {e}")
        if errors:
            self.settings_status_lbl.setText("Not saved: " + "; ".join(errors))
            self.settings_status_lbl.setStyleSheet("color:#c53030; font-size:9pt; font-weight:bold;")
            return
        changed = {k: v for k, v in new.items() if self.settings.get(k) != v}
        if not changed:
            self._settings_dirty = False
            self.settings_status_lbl.setText("No changes.")
            self.settings_status_lbl.setStyleSheet("color:#444; font-size:9pt;")
            return
        self._apply_settings_changes(dict(changed))
        self._populate_settings_tab()
        self.settings_status_lbl.setText(f"Saved {len(changed)} change(s): {', '.join(changed)}.")
        self.settings_status_lbl.setStyleSheet("color:#1b8a5a; font-size:9pt; font-weight:bold;")

    def _apply_settings_changes(self, changed: dict) -> None:
        """Store `changed` in self.settings, save settings.json, and push the
        new values to whatever already uses them."""
        new_wd = changed.pop("working_dir", None)
        new_cal = changed.pop("lightbox_campaign_path", None)
        self.settings.update(changed)
        save_settings(self.settings)
        if "iv_timeout_seconds" in changed:
            self.sweep_timeout_edit.setText(f"{self.settings['iv_timeout_seconds']:g}")
        if "panel_config_path" in changed or "reference_tolerance_pct" in changed:
            self._reload_panel_config()
        if new_wd is not None:
            if new_wd and not Path(new_wd).is_dir():
                self.set_status(f"Working folder {new_wd} does not exist; kept {self.working_dir}.", "warning")
            else:
                self._apply_working_dir(Path(new_wd) if new_wd else Path.cwd(), persist=True)
        if self.lightbox_mode:
            if new_cal is not None:
                if new_cal:
                    self._lightbox_load_calibration(new_cal)
                else:
                    self.settings["lightbox_campaign_path"] = ""
                    save_settings(self.settings)
            for key, combo in (("lightbox_irr_node", self.lb_irr_node_combo), ("lightbox_lux_node", self.lb_lux_node_combo)):
                if key in changed:
                    combo.setCurrentText(self.settings[key])
            self._lightbox_refresh_info()
        elif new_cal is not None:
            self.settings["lightbox_campaign_path"] = new_cal
            save_settings(self.settings)

    def _open_settings_dialog(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.settings.update(dlg.values())
            save_settings(self.settings)
            self.sweep_timeout_edit.setText(f"{self.settings['iv_timeout_seconds']:g}")
            # panel_config_path / reference_tolerance_pct may have changed.
            self._reload_panel_config()
            if self.lightbox_mode:
                self._lightbox_refresh_info()
            self._populate_settings_tab()

    def _commit_sweep_timeout(self) -> None:
        """Validate the Measure tab's Sweep Timeout box and persist it as
        iv_timeout_seconds. An invalid or non-positive entry reverts the box
        to the current setting rather than being silently used."""
        current = self.settings["iv_timeout_seconds"]
        try:
            value = float(self.sweep_timeout_edit.text().strip())
        except ValueError:
            value = None
        if value is None or value <= 0:
            self.sweep_timeout_edit.setText(f"{current:g}")
            self.set_status(f"Sweep timeout must be a positive number -- kept {current:g}s.", "warning")
            return
        if value != current:
            self.settings["iv_timeout_seconds"] = value
            save_settings(self.settings)
            self.set_status(f"Sweep timeout set to {value:g}s.", "info")
        self.sweep_timeout_edit.setText(f"{value:g}")

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
        "lightbox_setpoint_wm2", "panel_width_mm", "panel_height_mm", "vac_target", "vac_set",
        "irr_dark_offset", "irr_expected", "irr_deviation_pct", "lux_expected", "lux_deviation_pct",
        "irradiance_raw",
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
            offset, label = self._active_irr_offset()
            if offset is None:
                self.irr_value_lbl.setText(f"{irr_value:.4f} W/m²")
                self.irr_value_lbl.setToolTip("Raw sensor reading (no zero offset captured this session)")
            else:
                self.irr_value_lbl.setText(f"{irr_value - offset:.4f} W/m²")
                self.irr_value_lbl.setToolTip(f"Raw {irr_value:.4f} minus {offset:.4f} ({label})")
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

    async def _soft_reconnect_otii(self) -> bool:
        """Replace the Otii TCP connection with a fresh one, without
        restarting Otii. Returns False (and marks Otii disconnected) if the
        new connection fails."""
        async with self._otii_lock:
            self._close_otii_connection()
            try:
                connection, otii_object, active_proj, devices = await call_with_timeout(
                    connect_otii_session, timeout=self.settings["otii_call_timeout_seconds"]
                )
            except Exception as e:
                _log("warn", "otii_soft_reconnect_failed", error=str(e))
                self.otii_connection = None
                self.otii_object = None
                self.otii_project = None
                self.otii_devices = []
                self._set_otii_badge("● Otii: disconnected", "#ffcccc")
                self._reset_connect_button()
                return False
            self.otii_connection = connection
            self.otii_object = otii_object
            self.otii_project = active_proj
            self.otii_devices = devices
        _log("info", "otii_soft_reconnect_ok", devices=len(devices))
        self._set_otii_badge(f"● Otii: connected ({len(devices)} Arc)", "#a6ffcc")
        return True

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
        except Exception as e:
            # Not a hang but an error (e.g. a transaction-id mismatch after
            # an earlier timeout). Try a fresh connection.
            _log("warn", "otii_ping_failed", error=str(e))
            await self._soft_reconnect_otii()
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
        # Commit the Sweep Timeout box first: Ctrl+R doesn't move focus, so
        # an edit still sitting in the box wouldn't have fired
        # editingFinished yet.
        self._commit_sweep_timeout()
        lightbox_ctx = None
        if self.lightbox_mode:
            if self.otii_project is None or not self.otii_devices:
                self.set_status("No instrument connected", "error")
                return
            # All dialogs (setpoint, dark offset, next panel, light trim) run
            # here on the main thread, before anything is disabled.
            lightbox_ctx = self._lightbox_prepare_run()
            if lightbox_ctx is None:
                return
        self.btn_run.setEnabled(False)
        self.btn_connect.setEnabled(False)
        if not self._launch_task(self._async_start_measurement(lightbox_ctx)):
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

    async def _async_start_measurement(self, lightbox_ctx=None):
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
        delete_after = bool(self.settings.get("otii_delete_recordings", True))
        sweep_stats: dict = {}

        try:
            # Holds the same lock a connect/reconnect or an idle-watchdog
            # ping would hold before touching the Otii connection, so
            # neither can interleave with this measurement's own traffic on
            # the same TCP socket (2026-09 adversarial review finding #3).
            # Only wraps the actual Otii calls -- rendering/saving below
            # doesn't need the connection, so it's done outside the lock.
            async with self._otii_lock:
                isc_measured = await call_with_timeout(
                    short_circuit, self.otii_project, self.otii_devices,
                    timeout=max(call_timeout, 15.0), delete_recording_after=delete_after,
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
                        stats=sweep_stats,
                    )
                finally:
                    self._on_main(lambda: self.btn_end_sweep.setEnabled(False))
                    if sweep_stats:
                        _log("info", "iv_sweep_timing", panel=panel_name, **sweep_stats)

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
                if delete_after:
                    # The data is in hand (and saved to disk just below), so
                    # drop the recording from Otii. A project that keeps
                    # filling up is what made later sweeps slower.
                    try:
                        await call_with_timeout(delete_recording, recording, timeout=call_timeout)
                    except Exception as e:
                        _log("warn", "otii_delete_recording_failed", error=str(e))

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
            _log("warn", "measurement_failed", error=msg, error_type=type(e).__name__)
            # A failed or timed-out Otii call can leave a late reply sitting
            # on the socket (or an abandoned thread still reading it), which
            # then answers the next request with the wrong transaction id.
            # A fresh TCP connection clears that without restarting Otii.
            if not await self._soft_reconnect_otii():
                self.set_status(
                    f"Measurement failed: {msg}\nOtii then failed to reconnect. Click Connect to Otii.", "error"
                )
            self._on_main(_reenable_run_and_connect)
            return

        marker[1] = time.time()
        self._on_main(self._redraw_timeseries)
        self._measuring = False
        self._last_metrics = metrics
        self._last_lightbox_ctx = lightbox_ctx
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
        pending_offset = self._active_irr_offset()[0]
        if irr_value is not None and pending_offset is not None:
            irr_value -= pending_offset
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
        sweep_note = f" (sweep {sweep_stats['duration_s']:.0f} s)" if sweep_stats.get("duration_s") else ""
        self.set_status(f"Measurement complete{sweep_note} — review the plot, then Save This Test", "success")

    # ------------------------------------------------------------------
    # Lightbox mode (--lightbox): calibrated indoor lightbox, one setpoint
    # per round. Logic and dialogs live in lightbox_mode.py.
    # ------------------------------------------------------------------

    def _build_lightbox_group(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self._section_label("LIGHTBOX"))

        cal_row = QHBoxLayout()
        self.lb_cal_lbl = QLabel("Calibration: (none chosen)")
        self.lb_cal_lbl.setWordWrap(True)
        self.lb_cal_lbl.setStyleSheet("font-size:9pt;")
        cal_btn = QPushButton("Choose…")
        cal_btn.setToolTip("Choose the calibration campaign folder (contains campaign.json and captures.csv)")
        cal_btn.clicked.connect(self._guard(self._lightbox_choose_calibration))
        cal_row.addWidget(self.lb_cal_lbl, 1)
        cal_row.addWidget(cal_btn)
        lay.addLayout(cal_row)

        round_row = QHBoxLayout()
        self.lb_round_lbl = QLabel("Setpoint: (no round yet)")
        self.lb_round_lbl.setWordWrap(True)
        self.lb_round_lbl.setStyleSheet("font-size:10pt; font-weight:bold;")
        new_round_btn = QPushButton("New Round…")
        new_round_btn.setToolTip("Start a new round: new setpoint and a fresh dark-offset reading")
        new_round_btn.clicked.connect(self._guard(self._lightbox_new_round))
        round_row.addWidget(self.lb_round_lbl, 1)
        round_row.addWidget(new_round_btn)
        lay.addLayout(round_row)

        node_form = QFormLayout()
        labels = self._lightbox_node_labels()
        self.lb_irr_node_combo = QComboBox()
        self.lb_lux_node_combo = QComboBox()
        for combo, key in ((self.lb_irr_node_combo, "lightbox_irr_node"), (self.lb_lux_node_combo, "lightbox_lux_node")):
            combo.addItems(labels)
            combo.setCurrentText(self.settings.get(key) or "")
            combo.currentTextChanged.connect(self._guard1(self._on_lightbox_node_changed))
        node_form.addRow("Irradiance sensor node:", self.lb_irr_node_combo)
        node_form.addRow("Lux sensor node:", self.lb_lux_node_combo)
        lay.addLayout(node_form)

        self.lb_info_lbl = QLabel("")
        self.lb_info_lbl.setWordWrap(True)
        self.lb_info_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.lb_info_lbl.setStyleSheet("font-size:9pt; color:#333;")
        lay.addWidget(self.lb_info_lbl)
        return box

    def _lightbox_node_labels(self) -> list:
        if self.lb_calibration is not None:
            return self.lb_calibration.node_labels()
        return lbm.cm.all_node_labels(lbm.cm.GridConfig())

    def _lightbox_choose_calibration(self) -> bool:
        start = self.settings.get("lightbox_campaign_path") or str(PROJECT_ROOT / "calibration")
        chosen = QFileDialog.getExistingDirectory(self, "Choose Calibration Campaign Folder", start)
        if not chosen:
            return False
        return self._lightbox_load_calibration(chosen)

    def _lightbox_load_calibration(self, folder, quiet: bool = False) -> bool:
        """Load a calibration campaign folder. On failure the previous
        calibration (if any) is kept. A different folder ends the round,
        since the light setting came from the old calibration."""
        try:
            cal = lbm.Calibration(folder)
        except Exception as e:
            msg = f"Could not use {folder} as the lightbox calibration: {e}"
            self.set_status(msg + (" Keeping the previous calibration." if self.lb_calibration else ""), "warning")
            if not quiet:
                QMessageBox.warning(self, "Calibration not loaded", msg)
            self._lightbox_refresh_info()
            return False
        changed = self.lb_calibration is None or Path(self.lb_calibration.folder) != Path(cal.folder)
        self.lb_calibration = cal
        self.settings["lightbox_campaign_path"] = str(cal.folder)
        save_settings(self.settings)
        if changed and self.lb_round is not None:
            self.lb_round = None
            self.set_status(f"Calibration changed to {cal.summary}. Start a new round (setpoint + dark offset).", "warning")
        else:
            self.set_status(f"Lightbox calibration: {cal.summary}", "info")
        labels = cal.node_labels()
        for combo in (self.lb_irr_node_combo, self.lb_lux_node_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(labels)
            combo.setCurrentText(current if current in labels else labels[0])
            combo.blockSignals(False)
        self._lightbox_refresh_info()
        return True

    def _lightbox_new_round(self) -> bool:
        """Setpoint dialog, then the dark offset. Returns False if cancelled
        (the current round, if any, is left untouched)."""
        if self.lb_calibration is None and not self._lightbox_choose_calibration():
            return False
        last = self.lb_round.setpoint if self.lb_round else 1.0
        setpoint, ok = QInputDialog.getDouble(
            self, "New Round",
            "Setpoint: required average irradiance over the panel (W/m²).\n"
            "Every panel in this round is measured at this setpoint.",
            last, 0.001, 100000.0, 3,
        )
        if not ok:
            return False
        dlg = lbm.DarkOffsetDialog(
            self, self.irr_store.history, lbm.cr.IRR_ZERO_OFFSET_WM2, MIN_IRR_SAMPLES,
            stored_offset=(self.irr_zero_offset, self.irr_zero_offset_info) if self.irr_zero_offset is not None else None,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        if dlg.source == "measured":
            QMessageBox.information(self, "Dark offset recorded", "Uncover the irradiance sensor now.")
        self.lb_round = lbm.Round(setpoint=setpoint, dark_offset=dlg.offset, dark_offset_source=dlg.source)
        self.si_edit.setText(f"{setpoint:g}")
        self.set_status(
            f"New round: {setpoint:g} W/m², dark offset {dlg.offset:.4f} W/m² ({dlg.source}). Run Test to set up the first panel.",
            "info",
        )
        self._lightbox_refresh_info()
        return True

    def _on_lightbox_node_changed(self, _text) -> None:
        self.settings["lightbox_irr_node"] = self.lb_irr_node_combo.currentText()
        self.settings["lightbox_lux_node"] = self.lb_lux_node_combo.currentText()
        save_settings(self.settings)
        self._lightbox_refresh_info()

    def _lightbox_expected_now(self):
        """Expected readings at the current nodes for the round's current
        light setting, or None."""
        rd, cal = self.lb_round, self.lb_calibration
        if rd is None or rd.light is None or cal is None:
            return None
        w, h = rd.size
        try:
            return cal.expected(rd.light["vac_target"], self.lb_irr_node_combo.currentText(),
                                self.lb_lux_node_combo.currentText(), w, h)
        except ValueError:
            return None

    def _lightbox_refresh_info(self) -> None:
        if not self.lightbox_mode:
            return
        cal, rd = self.lb_calibration, self.lb_round
        self.lb_cal_lbl.setText(f"Calibration: {cal.summary}" if cal else "Calibration: (none chosen — click Choose…)")
        if rd is None:
            self.lb_round_lbl.setText("Setpoint: (no round yet — Run Test or New Round… starts one)")
            self.lb_info_lbl.setText("")
            return
        self.lb_round_lbl.setText(f"Setpoint: {rd.setpoint:g} W/m²   ·   dark offset {rd.dark_offset:.4f} W/m² ({rd.dark_offset_source})")
        if rd.light is None:
            self.lb_info_lbl.setText("Light not set yet for this round.")
            return
        L = rd.light
        w, h = rd.size
        lines = [
            f"Light set for <b>{w:g} × {h:g} mm</b> (last panel: {rd.last_panel or '—'})",
            f"Starting VAC {L['vac_target']:.1f} ± {L['vac_unc_v']:.2f} V · VAC as set {L['vac_set']:.1f}",
            f"Trimmed to {L['irr_target']:.4g} W/m² at {L['irr_node']}",
        ]
        exp = self._lightbox_expected_now()
        if exp is not None:
            ni, nl = exp["nodes"]["irr"], exp["nodes"]["lux"]
            lines.append(
                f"Expected: irr {ni['value']:.4g} W/m² at {ni['node']} · lux {nl['value']:.4g} lx at {nl['node']}"
                + (" · <span style='color:#8a5300'>a node is filled (glitch/excluded) in the calibration</span>"
                   if ni["glitch"] or nl["glitch"] else "")
            )
        self.lb_info_lbl.setText("<br>".join(lines))

    def _lightbox_live_irr(self):
        now = time.time()
        return lbm.window_mean(self.irr_store.history, now - lbm.TRIM_LIVE_WINDOW_S, now)

    def _lightbox_prepare_run(self):
        """Main-thread dialogs before a lightbox run. Returns the run context
        (everything the saved row needs) or None if cancelled/blocked."""
        if self.lb_calibration is None and not self._lightbox_choose_calibration():
            self.set_status("Choose a calibration folder before running in lightbox mode.", "warning")
            return None
        if self.lb_round is None and not self._lightbox_new_round():
            self.set_status("Run cancelled — no round started.", "info")
            return None
        cal, rd = self.lb_calibration, self.lb_round

        dlg = lbm.NextPanelDialog(
            self, self.panel_name_edit.text().strip(), self.lb_irr_node_combo.currentText(),
            self.lb_lux_node_combo.currentText(), cal.node_labels(),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.set_status("Run cancelled.", "info")
            return None
        panel_name, irr_node, lux_node = dlg.values()
        self.panel_name_edit.setText(panel_name)
        self.lb_irr_node_combo.setCurrentText(irr_node)
        self.lb_lux_node_combo.setCurrentText(lux_node)

        hit = norm_analysis.lookup_panel(panel_name, self.panel_config)
        if hit is None or hit[1].get("width_mm") is None or hit[1].get("height_mm") is None:
            what = "is not in the panel config" if hit is None else "has no Width/Height in the panel config"
            QMessageBox.warning(
                self, "Panel size needed",
                f"\"{panel_name}\" {what}. Add it with Width (along A–K) and Height on the Panel Config tab, then Run Test again.",
            )
            self.set_status(f"{panel_name}: add Width/Height on the Panel Config tab.", "warning")
            return None
        w, h = float(hit[1]["width_mm"]), float(hit[1]["height_mm"])
        if h > w:
            QMessageBox.warning(
                self, "Check orientation",
                f"\"{panel_name}\" is {w:g} × {h:g} mm: height is larger than width. The calibration assumes "
                "width along A–K is the longer side. Continuing with the sizes as configured.",
            )

        if rd.needs_light_setting(w, h):
            try:
                light = cal.light_setting(rd.setpoint, w, h, irr_node)
            except ValueError as e:
                QMessageBox.warning(self, "Setpoint not reachable", str(e))
                self.set_status(f"{panel_name}: {e}", "warning")
                return None
            trim = lbm.TrimDialog(
                self, panel_name, (w, h), rd.setpoint, light, rd.dark_offset,
                self.settings.get("lightbox_trim_tolerance_pct", 2.0), self._lightbox_live_irr,
            )
            if trim.exec() != QDialog.DialogCode.Accepted:
                self.set_status("Run cancelled — light not set.", "info")
                return None
            light["vac_set"] = trim.vac_set
            rd.size, rd.light = (w, h), light
            self.set_status(
                f"Light set for {w:g} × {h:g} mm: {light['vac_target']:.1f} VAC target, {trim.vac_set:.1f} VAC as set.", "info"
            )
        else:
            self.set_status(f"Same size as {rd.last_panel}: light unchanged ({rd.light['vac_set']:.1f} VAC).", "info")

        try:
            exp = cal.expected(rd.light["vac_target"], irr_node, lux_node, w, h)
        except ValueError as e:
            QMessageBox.warning(self, "Cannot predict sensor readings", str(e))
            return None
        rd.last_panel = panel_name
        self.si_edit.setText(f"{rd.setpoint:g}")
        self._lightbox_refresh_info()
        return {
            "setpoint": rd.setpoint, "w": w, "h": h, "dark_offset": rd.dark_offset,
            "vac_target": rd.light["vac_target"], "vac_set": rd.light["vac_set"],
            "irr_node": irr_node, "lux_node": lux_node,
            "irr_expected": exp["nodes"]["irr"]["value"], "lux_expected": exp["nodes"]["lux"]["value"],
            "campaign": cal.campaign_id if cal.campaign_id == cal.folder.name else f"{cal.campaign_id} ({cal.folder.name})",
        }

    def _lightbox_row_fields(self, ctx: dict, irr_mean, lux_mean):
        """Extra summary-CSV fields for a lightbox row, and a warning text if
        either sensor deviates by more than the threshold (else None)."""
        irr_meas = irr_mean - ctx["dark_offset"] if irr_mean is not None else None
        irr_dev = lbm.deviation_pct(irr_meas, ctx["irr_expected"])
        lux_dev = lbm.deviation_pct(lux_mean, ctx["lux_expected"])

        def num(v, nd=6):
            return "" if v is None else round(float(v), nd)

        extra = {
            "lightbox_setpoint_wm2": ctx["setpoint"], "panel_width_mm": ctx["w"], "panel_height_mm": ctx["h"],
            "vac_target": num(ctx["vac_target"], 2), "vac_set": ctx["vac_set"], "irr_dark_offset": num(ctx["dark_offset"]),
            "irr_node": ctx["irr_node"], "irr_expected": num(ctx["irr_expected"]), "irr_deviation_pct": num(irr_dev, 2),
            "lux_node": ctx["lux_node"], "lux_expected": num(ctx["lux_expected"], 4), "lux_deviation_pct": num(lux_dev, 2),
            "calibration_campaign": ctx["campaign"],
        }
        limit = self.settings.get("lightbox_deviation_warn_pct", 5.0)
        bad = [f"{name} at {node}: {dev:+.1f}%" for name, node, dev in
               (("Irradiance", ctx["irr_node"], irr_dev), ("Lux", ctx["lux_node"], lux_dev))
               if dev is not None and abs(dev) > limit]
        missing = [name for name, v in (("irradiance", irr_mean), ("lux", lux_mean)) if v is None]
        warning = None
        if bad or missing:
            parts = []
            if bad:
                parts.append(f"More than ±{limit:g}% from the calibration: " + "; ".join(bad) + ".")
            if missing:
                parts.append("No " + " or ".join(missing) + " readings during the sweep, so no deviation.")
            warning = " ".join(parts)
        return extra, warning

    def _lightbox_show_deviation(self, run_index, extra: dict, warning) -> None:
        def fmt(v):
            return "—" if v in ("", None) else f"{v:+.1f}%"
        line = (f"Deviation vs calibration: irr {fmt(extra['irr_deviation_pct'])} at {extra['irr_node']}, "
                f"lux {fmt(extra['lux_deviation_pct'])} at {extra['lux_node']}")
        self.results_lbl.setText(self.results_lbl.text() + "\n" + line)
        if warning:
            self.set_status(f"Saved run {run_index}. {warning}", "warning")
            QMessageBox.warning(self, "Sensor deviation", f"Run {run_index} was saved.\n\n{warning}")

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

        # Logged irradiance has the active offset subtracted (the lightbox
        # round's dark offset, else this session's zero offset); the raw mean
        # is kept in irradiance_raw. The stdev is unaffected by the offset.
        irr_raw = irr_mean
        irr_offset, _ = self._active_irr_offset()
        if self.lightbox_mode and self._last_lightbox_ctx is not None:
            irr_offset = self._last_lightbox_ctx["dark_offset"]
        if irr_mean is not None and irr_offset is not None:
            irr_mean = irr_raw - irr_offset
        irr_3sigma_pct = self._three_sigma_pct(irr_stdev, irr_mean)

        snapshot = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "lux": lux_mean if lux_mean is not None else 0.0,
            "irradiance": irr_mean if irr_mean is not None else 0.0,
            "samples": irr_count,
        }

        lightbox_extra, deviation_warning = None, None
        if self.lightbox_mode and self._last_lightbox_ctx is not None:
            lightbox_extra, deviation_warning = self._lightbox_row_fields(
                self._last_lightbox_ctx, irr_raw, lux_mean
            )
        row_extra = dict(lightbox_extra or {})
        if irr_offset is not None:
            row_extra["irr_dark_offset"] = round(float(irr_offset), 6)
            row_extra["irradiance_raw"] = irr_raw if irr_raw is not None else ""

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
            extra=row_extra,
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

        if lightbox_extra is not None:
            self._lightbox_show_deviation(run_index, lightbox_extra, deviation_warning)

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
        for _header, key, spec in _LIGHTBOX_COLUMNS:
            value = row.get(key, "")
            if spec and value not in ("", None):
                try:
                    value = format(float(value), spec)
                except (TypeError, ValueError):
                    pass
            values.append("" if value is None else value)
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


def install_exception_hooks(get_window) -> None:
    """PyQt6 aborts the whole process when an exception escapes any Python
    slot, but only while sys.excepthook is Python's default one. With a
    hook of our own installed, PyQt calls it and carries on. That covers
    every slot, including the ones not wrapped in _guard() (e.g. dialog
    buttons and timers). threading.excepthook does the same for background
    threads, which otherwise die with only a stderr trace."""
    def _report(exc_type, exc, tb, where):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        _log("error", "unhandled_exception", where=where, error=f"{exc_type.__name__}: {exc}")
        if sys.__stderr__ is not None:
            sys.__stderr__.write(text)
        window = get_window()
        if window is not None:
            try:
                window.set_status(f"Internal error (caught): {exc_type.__name__}: {exc}", "error")
            except Exception:
                pass

    def _excepthook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        _report(exc_type, exc, tb, "main")

    def _thread_excepthook(args):
        if args.exc_type is SystemExit:
            return
        _report(args.exc_type, args.exc_value, args.exc_traceback, getattr(args.thread, "name", "thread"))

    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook


def main():
    lightbox = "--lightbox" in sys.argv[1:]
    argv = [a for a in sys.argv if a != "--lightbox"]
    app = _SafeApplication(argv)
    apply_light_theme(app)
    holder = {"window": None}
    install_exception_hooks(lambda: holder["window"])
    window = LowLightApp(lightbox=lightbox)
    holder["window"] = window
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
