#!/usr/bin/env python3
import argparse
import ctypes
import json
import os
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

import serial

from triplett_lt68_probe_v3 import get_lux


IRRADIANCE_RE = re.compile(
    r"Irradiance:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


class RollingIrradiance:
    def __init__(self, window_seconds: float = 10.0):
        self.window_seconds = window_seconds
        self.samples = deque()
        self.lock = threading.Lock()

    def add(self, value: float) -> None:
        now = time.time()
        with self.lock:
            self.samples.append((now, value))
            self._prune(now)

    def average(self):
        now = time.time()
        with self.lock:
            self._prune(now)
            if not self.samples:
                return None, 0
            values = [v for _, v in self.samples]
            return sum(values) / len(values), len(values)

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


def has_gui_session() -> bool:
    if sys.platform.startswith("win"):
        session = os.environ.get("SESSIONNAME", "")
        return not session.lower().startswith("services")
    return True


def set_clipboard_windows(text: str) -> bool:
    CF_UNICODETEXT = 13
    GMEM_MOVEABLE = 0x0002
    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32

    if not user32.OpenClipboard(None):
        return False

    try:
        user32.EmptyClipboard()
        data = text.encode("utf-16-le")
        h_global = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data) + 2)
        if not h_global:
            return False

        locked = kernel32.GlobalLock(h_global)
        if not locked:
            kernel32.GlobalFree(h_global)
            return False

        ctypes.memmove(locked, data, len(data))
        ctypes.memset(locked + len(data), 0, 2)
        kernel32.GlobalUnlock(h_global)

        if not user32.SetClipboardData(CF_UNICODETEXT, h_global):
            kernel32.GlobalFree(h_global)
            return False

        return True
    finally:
        user32.CloseClipboard()


def serial_irradiance_reader(port: str, baud: int, store: RollingIrradiance, log) -> None:
    while True:
        try:
            log("info", "serial_opening", port=port, baud=baud)
            with serial.Serial(port, baudrate=baud, timeout=1) as ser:
                log("info", "serial_connected", port=port, baud=baud)
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
            log("error", "serial_error", error=str(e))
        except Exception as e:
            log("error", "serial_error", error=str(e))

        time.sleep(1.0)


def parse_hotkey(hotkey: str, keyboard):
    tokens = [t.strip().lower() for t in hotkey.split("+") if t.strip()]
    required_mods = []
    trigger_chars = set()
    trigger_key = None
    trigger_vks = set()

    modifier_map = {
        "ctrl": {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
        "shift": {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r},
        "alt": {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r},
    }
    vk_map = {
        "`": 192,
        "~": 192,
    }

    for token in tokens:
        if token in modifier_map:
            required_mods.append(modifier_map[token])
            continue
        if token in ("`", "grave", "backtick"):
            trigger_chars.update({"`", "~"})
            trigger_vks.add(192)
            continue
        if len(token) == 1:
            trigger_chars.add(token)
            if token in vk_map:
                trigger_vks.add(vk_map[token])
            elif token.isalnum():
                trigger_vks.add(ord(token.lower()))
            continue
        raise ValueError(f"Unsupported hotkey token: {token}")

    if not trigger_chars and trigger_key is None:
        raise ValueError("Hotkey must include a trigger key")

    return required_mods, trigger_chars, trigger_key, trigger_vks


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read lux + rolling irradiance and paste a row into Excel on hotkey trigger"
        )
    )
    parser.add_argument("--irr-port", required=True, help="Irradiance serial port, e.g. COM5")
    parser.add_argument("--irr-baud", type=int, default=115200, help="Irradiance serial baud")
    parser.add_argument("--lux-port", required=True, help="Lux meter serial port, e.g. COM6")
    parser.add_argument("--lux-timeout", type=float, default=1.0, help="Lux meter timeout in seconds")
    parser.add_argument("--lux-settle", type=float, default=0.25, help="Lux meter settle time in seconds")
    parser.add_argument("--window", type=float, default=10.0, help="Irradiance averaging window in seconds")
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
        "--newline",
        action="store_true",
        help="Append newline so Excel advances to the next row",
    )
    args = parser.parse_args()

    try:
        from pynput import keyboard
    except Exception as e:
        print(f"Missing dependency: pynput ({e})", file=sys.stderr)
        print("Install with: python -m pip install pynput", file=sys.stderr)
        return 2

    levels = {"debug": 10, "info": 20, "warn": 30, "error": 40}

    def should_log(level: str) -> bool:
        return levels[level] >= levels[args.log_level]

    def log(level: str, event: str, **fields) -> None:
        if should_log(level):
            log_event(level, event, **fields)

    store = RollingIrradiance(window_seconds=args.window)
    trigger_lock = threading.Lock()

    reader_thread = threading.Thread(
        target=serial_irradiance_reader,
        args=(args.irr_port, args.irr_baud, store, log),
        daemon=True,
    )
    reader_thread.start()

    try:
        required_mods, trigger_chars, trigger_key, trigger_vks = parse_hotkey(args.hotkey, keyboard)
    except ValueError as e:
        log("error", "hotkey_invalid", error=str(e), hotkey=args.hotkey)
        return 2

    def capture_row() -> None:
        with trigger_lock:
            avg_irr, count = store.average()
            if avg_irr is None:
                log("warn", "irradiance_unavailable", window=args.window)
                return

            try:
                lux = get_lux(port=args.lux_port, timeout=args.lux_timeout, settle=args.lux_settle)
            except Exception as e:
                log("error", "lux_read_failed", error=str(e), port=args.lux_port)
                return

            row = f"{lux:.3f}\t{avg_irr:.6f}"
            if args.newline:
                row += "\n"

            copied = set_clipboard_windows(row)
            if not copied:
                log("error", "clipboard_error", row=row)
                return

            log(
                "info",
                "row_copied",
                row=row.rstrip("\n"),
                lux=lux,
                irradiance=avg_irr,
                samples=count,
            )

            if args.auto_paste:
                delay_s = max(args.paste_delay_ms, 0) / 1000.0
                if delay_s:
                    time.sleep(delay_s)
                controller = keyboard.Controller()
                try:
                    with controller.pressed(keyboard.Key.ctrl):
                        controller.press("v")
                        controller.release("v")
                except Exception as e:
                    log("error", "paste_error", error=str(e))

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

        if mods_down and trigger_match and not hotkey_latched["active"]:
            hotkey_latched["active"] = True
            capture_row()

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

    log(
        "info",
        "startup",
        irr_port=args.irr_port,
        irr_baud=args.irr_baud,
        lux_port=args.lux_port,
        window=args.window,
        hotkey=args.hotkey,
    )

    if not has_gui_session():
        log("error", "hotkey_disabled", reason="no_gui_session")
        return 2

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
