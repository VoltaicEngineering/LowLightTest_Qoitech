import argparse
import csv
import json
import os
import re
import socket
import sys
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pyperclip
import serial

try:
    from pynput import keyboard
except Exception:
    keyboard = None

from triplett_lt68_probe_v3 import LT68, get_lux_snapshot


IRRADIANCE_RE = re.compile(
    r"Irradiance:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


class RollingIrradiance:
    def __init__(self, window_seconds: float = 10.0):
        self.window_seconds = window_seconds
        self.samples = deque()  # (timestamp, value)
        self.latest_value = None
        self.latest_ts = None
        self.lock = threading.Lock()

    def add(self, value: float) -> None:
        now = time.time()
        with self.lock:
            self.samples.append((now, value))
            self.latest_value = value
            self.latest_ts = now
            self._prune(now)

    def average(self):
        now = time.time()
        with self.lock:
            self._prune(now)
            if not self.samples:
                return None, 0
            values = [v for _, v in self.samples]
            return sum(values) / len(values), len(values)

    def latest(self):
        with self.lock:
            return self.latest_value, self.latest_ts

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()


def log_event(level: str, event: str, **fields) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
    }
    record.update(fields)
    print(json.dumps(record, separators=(",", ":")))


def serial_reader(
    port: str,
    baud: int,
    store: RollingIrradiance,
    max_retries: int,
    backoff_initial: float,
    backoff_multiplier: float,
    backoff_max: float,
    log,
) -> None:
    attempts = 0
    delay = backoff_initial

    while True:
        try:
            log("info", "serial_opening", port=port, baud=baud)
            with serial.Serial(port, baudrate=baud, timeout=1) as ser:
                log("info", "serial_connected", port=port, baud=baud)
                attempts = 0
                delay = backoff_initial
                while True:
                    raw = ser.readline()
                    if not raw:
                        continue

                    line = raw.decode("utf-8", errors="ignore").strip()
                    match = IRRADIANCE_RE.search(line)
                    if not match:
                        continue

                    value = float(match.group(1))
                    store.add(value)
                    log("debug", "sample_captured", irradiance=value)
        except serial.SerialException as e:
            attempts += 1
            log("error", "serial_error", error=str(e), attempt=attempts)
        except Exception as e:
            attempts += 1
            log("error", "serial_error", error=str(e), attempt=attempts)

        if attempts >= max_retries:
            log("error", "serial_giveup", attempts=attempts)
            return

        sleep_time = min(delay, backoff_max)
        log("warn", "retry_sleep", seconds=sleep_time, attempt=attempts)
        time.sleep(sleep_time)
        delay = min(delay * backoff_multiplier, backoff_max)


def parse_hotkey(hotkey: str):
    if keyboard is None:
        raise RuntimeError("pynput is unavailable; hotkeys cannot be enabled")

    tokens = [t.strip().lower() for t in hotkey.split("+") if t.strip()]
    required_mods = []
    trigger_chars = set()
    trigger_key = None
    trigger_vks = set()

    modifier_map = {
        "ctrl": {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
        "control": {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
        "shift": {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r},
        "alt": {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r},
        "option": {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r},
        "cmd": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
        "command": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
        "super": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
        "meta": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
        "win": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
    }

    special_keys = {
        "space": keyboard.Key.space,
        "tab": keyboard.Key.tab,
        "enter": keyboard.Key.enter,
        "return": keyboard.Key.enter,
        "esc": keyboard.Key.esc,
        "escape": keyboard.Key.esc,
    }

    vk_map = {
        "`": 192,
        "~": 192,
        "'": 222,
        '"': 222,
        ",": 188,
        "<": 188,
        ".": 190,
        ">": 190,
        "/": 191,
        "?": 191,
        ";": 186,
        ":": 186,
        "[": 219,
        "{": 219,
        "]": 221,
        "}": 221,
        "\\": 220,
        "|": 220,
        "-": 189,
        "_": 189,
        "=": 187,
        "+": 187,
    }

    for token in tokens:
        if token in modifier_map:
            required_mods.append(modifier_map[token])
            continue

        if token in ("`", "grave", "backtick"):
            trigger_chars.update({"`", "~"})
            trigger_vks.add(192)
            continue

        if token in ("'", 'quote'):
            trigger_chars.update({"'", '"'})
            trigger_vks.add(222)
            continue

        if token in ('"', 'doublequote'):
            trigger_chars.update({"'", '"'})
            trigger_vks.add(222)
            continue

        if token in special_keys:
            trigger_key = special_keys[token]
            continue

        if len(token) == 1:
            trigger_chars.add(token.lower())
            if token in vk_map:
                trigger_vks.add(vk_map[token])
            elif token.isalnum():
                trigger_vks.add(ord(token.lower()))
            continue

        raise ValueError(f"Unsupported hotkey token: {token}")

    if not trigger_chars and trigger_key is None:
        raise ValueError("Hotkey must include a trigger key")

    return required_mods, trigger_chars, trigger_key, trigger_vks


def has_gui_session() -> bool:
    platform = sys.platform
    if platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if platform == "darwin":
        if (os.environ.get("SSH_TTY") or os.environ.get("SSH_CONNECTION")) and not os.environ.get(
            "DISPLAY"
        ):
            return False
        return True

    if platform.startswith("win"):
        session = os.environ.get("SESSIONNAME", "")
        if session.lower().startswith("services"):
            return False
        return True

    return True


def run_tcp_server(host: str, port: int, on_request, log) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(5)
        log("info", "tcp_listening", host=host, port=port)

        while True:
            conn, addr = server.accept()
            with conn:
                client = f"{addr[0]}:{addr[1]}"
                try:
                    raw = conn.recv(4096)
                    command = raw.decode("utf-8", errors="ignore").strip().upper() if raw else "TRIGGER"
                    if not command:
                        command = "TRIGGER"
                    log("info", "tcp_request", client=client, command=command)
                    response = on_request(command)
                except Exception as e:
                    response = {"ok": False, "error": str(e)}
                    log("error", "tcp_request_failed", client=client, error=str(e))

                payload = json.dumps(response, separators=(",", ":")) + "\n"
                conn.sendall(payload.encode("utf-8"))


def send_paste(log) -> bool:
    if keyboard is None:
        log("warn", "paste_error", method="pynput", error="pynput unavailable")
        return False

    controller = keyboard.Controller()
    platform = sys.platform

    if platform == "darwin":
        modifier = keyboard.Key.cmd
    else:
        modifier = keyboard.Key.ctrl

    try:
        with controller.pressed(modifier):
            controller.press("v")
            controller.release("v")
        log("debug", "paste_sent", method="pynput")
        return True
    except Exception as e:
        log("warn", "paste_error", method="pynput", error=str(e))

    if platform.startswith("linux"):
        try:
            subprocess.run(["xdotool", "key", "ctrl+v"], check=True)
            log("info", "paste_fallback_used", method="xdotool")
            return True
        except Exception as e:
            log("error", "paste_error", method="xdotool", error=str(e))

    return False


def set_clipboard(text: str, log) -> bool:
    try:
        pyperclip.copy(text)
        return True
    except Exception as e:
        log("warn", "clipboard_copy_failed", error=str(e), method="pyperclip")

    if sys.platform.startswith("win"):
        try:
            import ctypes
            import ctypes.wintypes as wintypes

            CF_UNICODETEXT = 13
            kernel32 = ctypes.windll.kernel32
            user32 = ctypes.windll.user32

            if not user32.OpenClipboard(None):
                raise RuntimeError("OpenClipboard failed")
            user32.EmptyClipboard()

            data = text.encode("utf-16-le")
            h_global = kernel32.GlobalAlloc(0x0002, len(data) + 2)
            if not h_global:
                raise RuntimeError("GlobalAlloc failed")

            locked = kernel32.GlobalLock(h_global)
            if not locked:
                kernel32.GlobalFree(h_global)
                raise RuntimeError("GlobalLock failed")

            ctypes.memmove(locked, data, len(data))
            kernel32.GlobalUnlock(h_global)
            if not user32.SetClipboardData(CF_UNICODETEXT, h_global):
                kernel32.GlobalFree(h_global)
                raise RuntimeError("SetClipboardData failed")

            user32.CloseClipboard()
            log("info", "clipboard_copy_used", method="win32")
            return True
        except Exception as e:
            log("error", "clipboard_copy_failed", method="win32", error=str(e))

    return False


CSV_HEADERS = [
    "ts_local_iso",
    "ts_utc_iso",
    "elapsed_s",
    "lux",
    "irradiance",
    "irr_age_s",
    "lux_ok",
    "irr_ok",
    "error",
]


class HourlyCsvLogger:
    def __init__(self, log_dir: str, run_id: str, flush_every: int, log):
        self.log_dir = Path(log_dir)
        self.run_id = run_id
        self.flush_every = max(1, flush_every)
        self.log = log

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_hour_key = None
        self.hour_index = 0
        self.files_created = 0
        self.rows_written = 0

        self._file = None
        self._writer = None
        self.current_path = None

    def _open_unique_file(self, base_name: str):
        for suffix in range(1, 10000):
            name = f"{base_name}.csv" if suffix == 1 else f"{base_name}_v{suffix}.csv"
            path = self.log_dir / name
            try:
                file_obj = path.open("x", newline="", encoding="utf-8")
                return path, file_obj
            except FileExistsError:
                continue
        raise RuntimeError("could not create a unique hourly csv filename")

    def _rotate_if_needed(self, now_local: datetime) -> None:
        hour_key = now_local.strftime("%Y%m%d_%H")
        if self._writer is not None and hour_key == self.current_hour_key:
            return

        self.close()
        base_name = f"lux_irr_{self.run_id}_h{self.hour_index:02d}_{hour_key}"
        path, file_obj = self._open_unique_file(base_name)

        self._file = file_obj
        self._writer = csv.writer(file_obj)
        self._writer.writerow(CSV_HEADERS)
        self._file.flush()

        self.current_hour_key = hour_key
        self.current_path = str(path)
        self.hour_index += 1
        self.files_created += 1

        self.log("info", "csv_rotated", path=self.current_path, hour_key=hour_key)

    def write_row(self, now_local: datetime, row: list) -> None:
        self._rotate_if_needed(now_local)
        if self._writer is None or self._file is None:
            raise RuntimeError("csv writer not initialized")
        self._writer.writerow(row)
        self.rows_written += 1
        if self.rows_written % self.flush_every == 0:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
            finally:
                self._file.close()
        self._file = None
        self._writer = None


def run_periodic_logger(args, store: RollingIrradiance, log) -> int:
    if not args.lux_port:
        log("error", "lux_port_missing", message="--lux-port is required with --log-dir")
        return 2

    run_started_local = datetime.now()
    run_id = args.run_name or run_started_local.strftime("run-%Y%m%d-%H%M%S")
    duration_seconds = None if args.duration_hours is None else max(0.0, args.duration_hours * 3600.0)
    interval = max(0.1, args.interval_s)
    stale_threshold = max(0.0, args.stale_irr_s)

    writer = HourlyCsvLogger(args.log_dir, run_id, args.flush_every, log)
    meter = None

    start_mono = time.monotonic()
    deadline_mono = None if duration_seconds is None else start_mono + duration_seconds
    next_tick = start_mono
    next_heartbeat = start_mono + 60.0

    lux_failures = 0
    irr_stale = 0
    reconnect_delay = 1.0
    reconnect_after_mono = 0.0

    log(
        "info",
        "periodic_logger_started",
        log_dir=args.log_dir,
        run_id=run_id,
        interval_s=interval,
        duration_hours=args.duration_hours,
    )

    try:
        while True:
            now_mono = time.monotonic()
            if deadline_mono is not None and now_mono >= deadline_mono:
                break

            sleep_time = next_tick - now_mono
            if sleep_time > 0:
                time.sleep(sleep_time)

            sample_mono = time.monotonic()
            sample_epoch = time.time()
            now_local = datetime.now()
            now_utc = datetime.now(timezone.utc)

            irr_value, irr_ts = store.latest()
            irr_age_s = None if irr_ts is None else max(0.0, sample_epoch - irr_ts)
            irr_ok = irr_value is not None and irr_age_s is not None and irr_age_s <= stale_threshold
            if not irr_ok:
                irr_stale += 1

            errors = []

            lux_value = float("nan")
            lux_ok = False
            if meter is None and sample_mono >= reconnect_after_mono:
                try:
                    meter = LT68(port=args.lux_port, timeout=args.lux_timeout)
                    reconnect_delay = 1.0
                    log("info", "lux_connected", port=args.lux_port)
                except Exception as e:
                    lux_failures += 1
                    errors.append(f"lux_connect_failed:{e}")
                    reconnect_after_mono = sample_mono + min(reconnect_delay, 15.0)
                    reconnect_delay = min(reconnect_delay * 2.0, 15.0)
            elif meter is None:
                errors.append("lux_reconnect_backoff")

            if meter is not None:
                try:
                    lux_value = meter.get_lux(settle=args.lux_settle)
                    lux_ok = True
                except Exception as e:
                    lux_failures += 1
                    errors.append(f"lux_read_failed:{e}")
                    try:
                        meter.close()
                    except Exception:
                        pass
                    meter = None
                    reconnect_after_mono = sample_mono + min(reconnect_delay, 15.0)
                    reconnect_delay = min(reconnect_delay * 2.0, 15.0)

            irradiance_out = irr_value if irr_value is not None else float("nan")
            error_text = ";".join(errors)

            writer.write_row(
                now_local,
                [
                    now_local.isoformat(timespec="seconds"),
                    now_utc.isoformat(),
                    round(sample_mono - start_mono, 3),
                    lux_value,
                    irradiance_out,
                    "" if irr_age_s is None else round(irr_age_s, 3),
                    int(lux_ok),
                    int(irr_ok),
                    error_text,
                ],
            )

            if sample_mono >= next_heartbeat:
                log(
                    "info",
                    "periodic_heartbeat",
                    rows_written=writer.rows_written,
                    files_created=writer.files_created,
                    current_file=writer.current_path,
                    lux_failures=lux_failures,
                    irr_stale=irr_stale,
                )
                next_heartbeat += 60.0

            next_tick += interval
            while next_tick <= sample_mono:
                next_tick += interval

    except KeyboardInterrupt:
        log("info", "periodic_logger_interrupt")
    finally:
        if meter is not None:
            try:
                meter.close()
            except Exception:
                pass
        writer.close()

    log(
        "info",
        "periodic_logger_stopped",
        rows_written=writer.rows_written,
        files_created=writer.files_created,
        lux_failures=lux_failures,
        irr_stale=irr_stale,
    )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Listen to serial Irradiance values and copy 10-second average via hotkey or TCP"
        )
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--window", type=float, default=10.0, help="Averaging window in seconds")
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=8765,
        help="Local TCP port for trigger/request API (default: 8765)",
    )
    parser.add_argument(
        "--lux-port",
        default=None,
        help="Lux meter serial port, e.g. COM6. Required for SNAPSHOT/COPY row mode",
    )
    parser.add_argument(
        "--lux-timeout",
        type=float,
        default=1.0,
        help="Lux meter timeout in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--lux-settle",
        type=float,
        default=0.25,
        help="Lux meter settle time in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum irradiance samples required for snapshot (default: 3)",
    )
    parser.add_argument(
        "--hotkey",
        default="ctrl+shift+`",
        help='Hotkey string (default: "ctrl+shift+`")',
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warn", "error"],
        help="Log level (default: info)",
    )
    parser.add_argument(
        "--auto-paste",
        dest="auto_paste",
        action="store_true",
        default=True,
        help="Auto-paste after hotkey trigger (default: on)",
    )
    parser.add_argument(
        "--no-auto-paste",
        dest="auto_paste",
        action="store_false",
        help="Disable auto-paste after hotkey trigger",
    )
    parser.add_argument(
        "--paste-delay-ms",
        type=int,
        default=10,
        help="Delay before auto-paste in milliseconds (default: 10)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Enable periodic logging mode and write hourly CSV files into this directory",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=1.0,
        help="Periodic logging interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=20.0,
        help="Periodic logging duration in hours (default: 20.0)",
    )
    parser.add_argument(
        "--stale-irr-s",
        type=float,
        default=3.0,
        help="Mark irradiance stale when latest sample age exceeds this threshold",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Flush CSV every N rows (default: 1)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run identifier used in CSV filenames",
    )
    args = parser.parse_args()

    store = RollingIrradiance(window_seconds=args.window)
    trigger_lock = threading.Lock()

    def should_log(level: str) -> bool:
        levels = {"debug": 10, "info": 20, "warn": 30, "error": 40}
        return levels[level] >= levels[args.log_level]

    def log(level: str, event: str, **fields) -> None:
        if should_log(level):
            log_event(level, event, **fields)

    log(
        "info",
        "startup",
        serial_port=args.port,
        baud=args.baud,
        window=args.window,
        lux_port=args.lux_port,
        tcp_port=args.tcp_port,
    )

    if hasattr(pyperclip, "is_available") and not pyperclip.is_available():
        log(
            "warn",
            "clipboard_unavailable",
            message=(
                "Pyperclip has no copy/paste mechanism. On Linux, install xclip or xsel "
                "(X11) or wl-clipboard (Wayland)."
            ),
        )

    thread = threading.Thread(
        target=serial_reader,
        args=(
            args.port,
            args.baud,
            store,
            5,
            1.0,
            2.0,
            30.0,
            log,
        ),
        daemon=True,
    )
    thread.start()

    if args.log_dir:
        return run_periodic_logger(args, store, log)

    def capture_snapshot(include_lux: bool = True) -> dict:
        with trigger_lock:
            avg, count = store.average()

            if avg is None:
                return {
                    "ok": False,
                    "error": "irradiance_unavailable",
                    "window_seconds": args.window,
                }

            if count < args.min_samples:
                return {
                    "ok": False,
                    "error": "insufficient_irradiance_samples",
                    "samples": count,
                    "minimum_required": args.min_samples,
                }

            lux_payload = None
            if include_lux:
                if not args.lux_port:
                    return {
                        "ok": False,
                        "error": "lux_port_missing",
                    }
                try:
                    lux_payload = get_lux_snapshot(
                        port=args.lux_port,
                        timeout=args.lux_timeout,
                        settle=args.lux_settle,
                    )
                except Exception as e:
                    return {
                        "ok": False,
                        "error": "lux_read_failed",
                        "detail": str(e),
                    }

            snapshot = {
                "ok": True,
                "ts": datetime.now(timezone.utc).isoformat(),
                "irradiance": avg,
                "samples": count,
            }
            if lux_payload is not None:
                snapshot["lux"] = lux_payload["lux"]
                snapshot["lux_meta"] = lux_payload
            return snapshot

    def copy_snapshot_row(auto_paste: bool) -> dict:
        snapshot = capture_snapshot(include_lux=bool(args.lux_port))
        if not snapshot.get("ok"):
            log("warn", "snapshot_failed", **snapshot)
            return snapshot

        if "lux" in snapshot:
            text = f"{snapshot['lux']:.3f}\t{snapshot['irradiance']:.6f}"
        else:
            text = f"{snapshot['irradiance']:.6f}"

        copied = set_clipboard(text, log)
        snapshot["clipboard"] = text

        if copied:
            log(
                "info",
                "snapshot_copied",
                value=text,
                samples=snapshot["samples"],
                window=args.window,
            )
            if auto_paste:
                delay_s = max(args.paste_delay_ms, 0) / 1000.0
                if delay_s:
                    time.sleep(delay_s)
                send_paste(log)
        else:
            snapshot["ok"] = False
            snapshot["error"] = "clipboard_error"
            log("error", "clipboard_error", value=text, samples=snapshot["samples"])

        return snapshot

    def handle_tcp_command(command: str) -> dict:
        if command in {"TRIGGER", "AVG", "AVERAGE"}:
            avg, count = store.average()
            if avg is None:
                return {"ok": False, "error": "irradiance_unavailable", "window_seconds": args.window}
            return {"ok": True, "irradiance": avg, "samples": count}

        if command in {"SNAPSHOT", "MEASURE"}:
            return capture_snapshot(include_lux=True)

        if command == "COPY":
            return copy_snapshot_row(auto_paste=False)

        return {
            "ok": False,
            "error": "unknown_command",
            "supported": ["TRIGGER", "SNAPSHOT", "COPY"],
        }

    tcp_thread = threading.Thread(
        target=run_tcp_server,
        args=("127.0.0.1", args.tcp_port, handle_tcp_command, log),
        daemon=True,
    )
    tcp_thread.start()

    try:
        required_mods, trigger_chars, trigger_key, trigger_vks = parse_hotkey(args.hotkey)
    except (ValueError, RuntimeError) as e:
        log("error", "hotkey_invalid", error=str(e), hotkey=args.hotkey)
        required_mods, trigger_chars, trigger_key, trigger_vks = [], set(), None, set()

    pressed = set()
    hotkey_latched = {"active": False}

    def _key_char(key):
        char = getattr(key, "char", None)
        if isinstance(char, str):
            return char.lower()
        return None

    def on_press(key):
        pressed.add(key)
        char_lower = _key_char(key)
        mods_down = all(any(k in pressed for k in modset) for modset in required_mods)
        trigger_match = False

        key_vk = getattr(key, "vk", None)
        if trigger_key is not None and key == trigger_key:
            trigger_match = True
        elif key_vk is not None and key_vk in trigger_vks:
            trigger_match = True
        elif char_lower is not None and char_lower in trigger_chars:
            trigger_match = True

        log("debug", "hotkey_key_state", key=str(key), char=char_lower, vk=key_vk, mods_down=mods_down, trigger_match=trigger_match)

        if mods_down and trigger_match and not hotkey_latched["active"]:
            hotkey_latched["active"] = True
            log(
                "debug",
                "hotkey_triggered",
                hotkey=args.hotkey,
                char=char_lower,
                key=str(key),
            )
            copy_snapshot_row(args.auto_paste)

    def on_release(key):
        pressed.discard(key)

        key_vk = getattr(key, "vk", None)
        if key == trigger_key or key_vk in trigger_vks:
            hotkey_latched["active"] = False
            return

        char_lower = _key_char(key)
        if char_lower in trigger_chars:
            hotkey_latched["active"] = False

        if any(key in modset for modset in required_mods):
            hotkey_latched["active"] = False

    if required_mods or trigger_chars or trigger_key is not None:
        if keyboard is None:
            log("warn", "hotkey_disabled", reason="pynput_unavailable")
            while True:
                time.sleep(1)
        elif has_gui_session():
            log("info", "hotkey_enabled", hotkey=args.hotkey)
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()
        else:
            log("warn", "hotkey_disabled", reason="no_gui_session")
            while True:
                time.sleep(1)
    else:
        log("warn", "hotkey_disabled", reason="invalid_hotkey")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
