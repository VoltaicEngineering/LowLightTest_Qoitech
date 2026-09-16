"""Shared sensor layer for the lux meter and irradiance sensor.

Extracted verbatim from low_light_app.py (2026-09, Phase 0 of the lightbox
calibration build — see docs/lightbox-calibration-plan.md) so a second app
(lightbox_calibration.py) can talk to the same hardware without importing
the whole Qt measurement application. low_light_app.py re-imports everything
here; its behaviour is unchanged.
"""
from __future__ import annotations

import os
import re
import threading
import time
from collections import deque
from pathlib import Path

import serial
import serial.tools.list_ports

from triplett_lt68_probe_v3 import LT68
from listen import RollingIrradiance, serial_reader, log_event

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "cache"

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


def _log(level: str, event: str, **fields) -> None:
    log_event(level, event, **fields)


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


def port_in_use(port: str) -> bool:
    """Best-effort check for whether a COM port is already held by another
    process. Windows serial ports are exclusive, so low_light_app.py and
    lightbox_calibration.py cannot both hold the same port; opening (and
    immediately closing) it is the only reliable way to find out short of
    parsing OS-specific device state."""
    try:
        with serial.Serial(port, timeout=0.1):
            pass
    except serial.SerialException:
        return True
    except Exception:
        return False
    return False


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
