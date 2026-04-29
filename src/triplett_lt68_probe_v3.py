#!/usr/bin/env python3
"""
Triplett LT68 / Extech HD450 / PCE-174 probe script.

Adds a simple Python API:
    lux = get_lux("COM6")

The helper forces lux units and autoranges the meter before returning a value.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import serial

MAGIC_SEND = b"\x87\x83"
CMD_READ_LIVE = 0x11
CMD_UNITS = 0xFE
CMD_RANGE = 0x7F

# Protocol reports two decimal-digit bytes for the displayed value and two more
# for the raw value. The instrument family uses a 4000-count display, so values
# above 3999 counts are treated as overload/OL.
MAX_DISPLAY_COUNTS = 3999
FC_TO_LUX = 10.76

RANGE_CYCLE = {
    "lux": ("400k", "400", "4k", "40k"),
    "fc": ("40k", "40", "400", "4k"),
}

RANGE_FACTOR = {
    "40": 0.01,
    "400": 0.1,
    "4k": 1.0,
    "40k": 10.0,
    "400k": 100.0,
}

LUX_ASCENDING = ("400", "4k", "40k", "400k")
LUX_MAX = {name: MAX_DISPLAY_COUNTS * RANGE_FACTOR[name] for name in LUX_ASCENDING}


class OverRangeError(RuntimeError):
    """Reading exceeds the meter's top lux range."""


def bcd_to_int(b: int) -> int:
    hi = (b >> 4) & 0x0F
    lo = b & 0x0F
    if hi > 9 or lo > 9:
        raise ValueError(f"invalid BCD byte: 0x{b:02X}")
    return hi * 10 + lo


def range_from_stat0(stat0: int, unit: str) -> tuple[str, float]:
    level = stat0 & 0b11
    return {
        "lux": {
            0: ("400k", 100.0),
            1: ("400", 0.1),
            2: ("4k", 1.0),
            3: ("40k", 10.0),
        },
        "fc": {
            0: ("40k", 10.0),
            1: ("40", 0.01),
            2: ("400", 0.1),
            3: ("4k", 1.0),
        },
    }[unit][level]


def decode_mode(stat0: int) -> str:
    mode_bits = (stat0 >> 3) & 0b111
    return {
        0b000: "normal",
        0b010: "Pmin",
        0b011: "Pmax",
        0b100: "max",
        0b101: "min",
        0b110: "rel",
    }.get(mode_bits, f"unknown({mode_bits:03b})")


def decode_view(stat1: int) -> str:
    view_bits = (stat1 >> 2) & 0b11
    return {
        0b00: "time",
        0b01: "day",
        0b10: "sampling",
        0b11: "year",
    }[view_bits]


def decode_memstat(stat1: int) -> str:
    mem_bits = stat1 & 0b11
    return {
        0b00: "none",
        0b01: "store",
        0b10: "recall",
        0b11: "logging",
    }[mem_bits]


@dataclass
class LiveReading:
    year: int
    month: int
    day: int
    weekday: int
    hour: int
    minute: int
    second: int
    value: float
    raw_value: float
    display_counts: int
    raw_counts: int
    unit: str
    range_name: str
    mode: str
    hold: str
    apo: str
    power: str
    view: str
    memstat: str
    mem_no: int
    read_no: int
    stat0: int
    stat1: int

    @property
    def iso_datetime(self) -> str:
        return (
            f"{self.year:04d}-{self.month:02d}-{self.day:02d}"
            f"T{self.hour:02d}:{self.minute:02d}:{self.second:02d}"
        )

    @property
    def overloaded(self) -> bool:
        return abs(self.display_counts) > MAX_DISPLAY_COUNTS or abs(self.raw_counts) > MAX_DISPLAY_COUNTS

    @property
    def lux_value(self) -> float:
        value = self.raw_value if self.mode == "rel" else self.value
        if self.unit == "lux":
            return value
        return value * FC_TO_LUX


class LT68:
    def __init__(self, port: str, timeout: float = 1.0):
        self.port = port
        self.ser = serial.Serial(
            port=port,
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def __enter__(self) -> "LT68":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self.ser.is_open:
            self.ser.close()

    def send_cmd(self, code: int) -> None:
        self.ser.reset_input_buffer()
        self.ser.write(MAGIC_SEND + bytes([code]))
        self.ser.flush()

    def press(self, name: str, count: int = 1, settle: float = 0.20) -> None:
        code = {
            "units": CMD_UNITS,
            "range": CMD_RANGE,
        }[name]
        for i in range(count):
            self.send_cmd(code)
            if i != count - 1 or settle > 0:
                time.sleep(settle)

    def read_live_raw(self) -> bytes:
        """
        Read one live-data packet and resynchronize on the AA DD header.

        On some LT68/HD450-family units the response stream can include one or
        more leading junk/control bytes before the documented AA DD magic.  The
        upstream reverse-engineering notes also warn that the vendor protocol
        documentation is partially incorrect, so this reader does not assume the
        packet starts at byte 0.
        """
        self.send_cmd(CMD_READ_LIVE)

        target_len = 18
        header = b"\xAA\xDD"
        timeout = self.ser.timeout if self.ser.timeout is not None else 1.0
        deadline = time.monotonic() + max(timeout, 1.0)
        buf = bytearray()

        while time.monotonic() < deadline:
            chunk = self.ser.read(1)
            if not chunk:
                continue
            buf.extend(chunk)

            idx = buf.find(header)
            if idx == -1:
                if len(buf) > 64:
                    del buf[:-2]
                continue

            if idx > 0:
                del buf[:idx]

            while len(buf) < target_len and time.monotonic() < deadline:
                chunk = self.ser.read(target_len - len(buf))
                if not chunk:
                    continue
                buf.extend(chunk)

            if len(buf) >= target_len and buf[:2] == header:
                return bytes(buf[:target_len])

        raise TimeoutError(
            "could not read a full live packet after syncing to AA DD; "
            f"partial bytes: {bytes(buf).hex(' ')}"
        )

    def read_live(self) -> LiveReading:
        data = self.read_live_raw()

        year = 2000 + bcd_to_int(data[3])
        weekday = bcd_to_int(data[4])
        month = bcd_to_int(data[5])
        day = bcd_to_int(data[6])
        hour = bcd_to_int(data[7])
        minute = bcd_to_int(data[8])
        second = bcd_to_int(data[9])

        stat0 = data[14]
        stat1 = data[15]

        unit = "fc" if ((stat0 >> 2) & 0b1) else "lux"
        range_name, factor = range_from_stat0(stat0, unit)

        sign = -1 if ((stat1 >> 4) & 0b1) else 1
        display_counts = sign * (100 * data[10] + data[11])
        raw_counts = sign * (100 * data[12] + data[13])
        value = display_counts * factor
        raw_value = raw_counts * factor

        return LiveReading(
            year=year,
            month=month,
            day=day,
            weekday=weekday,
            hour=hour,
            minute=minute,
            second=second,
            value=value,
            raw_value=raw_value,
            display_counts=display_counts,
            raw_counts=raw_counts,
            unit=unit,
            range_name=range_name,
            mode=decode_mode(stat0),
            hold="hold" if ((stat0 >> 6) & 0b1) else "cont",
            apo="off" if ((stat0 >> 7) & 0b1) else "on",
            power="low" if ((stat1 >> 5) & 0b1) else "ok",
            view=decode_view(stat1),
            memstat=decode_memstat(stat1),
            mem_no=data[16],
            read_no=data[17],
            stat0=stat0,
            stat1=stat1,
        )

    def ensure_lux(self, settle: float = 0.25) -> LiveReading:
        r = self.read_live()
        if r.unit != "lux":
            self.press("units", settle=settle)
            r = self.read_live()
        return r

    def set_range(self, target: str, settle: float = 0.25) -> LiveReading:
        r = self.read_live()
        cycle = RANGE_CYCLE[r.unit]
        if target not in cycle:
            raise ValueError(f"target range {target!r} invalid for unit {r.unit!r}")
        current_idx = cycle.index(r.range_name)
        target_idx = cycle.index(target)
        presses = (target_idx - current_idx) % len(cycle)
        if presses:
            self.press("range", count=presses, settle=settle)
        time.sleep(settle)
        return self.read_live()

    def _next_higher_lux_range(self, range_name: str) -> str:
        idx = LUX_ASCENDING.index(range_name)
        return LUX_ASCENDING[min(idx + 1, len(LUX_ASCENDING) - 1)]

    def _maybe_better_lower_lux_range(self, lux_value: float, current_range: str) -> str:
        idx = LUX_ASCENDING.index(current_range)
        while idx > 0:
            smaller = LUX_ASCENDING[idx - 1]
            # Add some headroom to avoid constant bouncing between ranges.
            if lux_value <= 0.90 * LUX_MAX[smaller]:
                idx -= 1
            else:
                break
        return LUX_ASCENDING[idx]

    def get_lux_reading(self, settle: float = 0.25, max_steps: int = 8) -> LiveReading:
        """
        Return a live reading in lux after forcing lux units and autoranging.

        Overload detection is inferred from the instrument's 4000-count display:
        when the protocol returns more than 3999 counts in either live value field,
        the reading is treated as over-range and the code steps to the next higher
        range before retrying.
        """
        reading = self.ensure_lux(settle=settle)

        for _ in range(max_steps):
            if reading.hold == "hold":
                raise RuntimeError("meter is in HOLD mode; release hold before calling get_lux()")
            if reading.mode in {"max", "min", "Pmax", "Pmin"}:
                raise RuntimeError(
                    f"meter is in {reading.mode} mode; switch back to normal mode before calling get_lux()"
                )

            if reading.overloaded:
                if reading.range_name == "400k":
                    raise OverRangeError("reading exceeds the LT68 top range (400k lux)")
                reading = self.set_range(self._next_higher_lux_range(reading.range_name), settle=settle)
                continue

            target = self._maybe_better_lower_lux_range(abs(reading.lux_value), reading.range_name)
            if target != reading.range_name:
                reading = self.set_range(target, settle=settle)
                continue

            final = self.read_live()
            if final.unit != "lux":
                raise RuntimeError("meter unit changed unexpectedly during autorange")
            if final.overloaded and final.range_name != "400k":
                reading = self.set_range(self._next_higher_lux_range(final.range_name), settle=settle)
                continue
            return final

        raise RuntimeError("autorange did not converge")

    def get_lux(self, settle: float = 0.25, max_steps: int = 8) -> float:
        return self.get_lux_reading(settle=settle, max_steps=max_steps).lux_value


def get_lux(port: str = "COM6", timeout: float = 1.0, settle: float = 0.25) -> float:
    """Simple one-shot helper for scripts."""
    with LT68(port=port, timeout=timeout) as meter:
        return meter.get_lux(settle=settle)


def get_lux_snapshot(port: str = "COM6", timeout: float = 1.0, settle: float = 0.25) -> dict:
    """Return a one-shot lux reading with useful metadata for integrations."""
    with LT68(port=port, timeout=timeout) as meter:
        reading = meter.get_lux_reading(settle=settle)
    return {
        "iso_datetime": reading.iso_datetime,
        "lux": reading.lux_value,
        "range": reading.range_name,
        "overloaded": reading.overloaded,
        "unit": reading.unit,
        "mode": reading.mode,
    }


def print_human(r: LiveReading) -> None:
    print(f"time      : {r.iso_datetime}")
    print(f"value     : {r.value} {r.unit}")
    print(f"raw_value : {r.raw_value} {r.unit}")
    print(f"lux_value : {r.lux_value} lux")
    print(f"range     : {r.range_name}")
    print(f"overload  : {r.overloaded}")
    print(f"mode      : {r.mode}")
    print(f"hold      : {r.hold}")
    print(f"apo       : {r.apo}")
    print(f"power     : {r.power}")
    print(f"view      : {r.view}")
    print(f"memstat   : {r.memstat}")
    print(f"mem_no    : {r.mem_no}")
    print(f"read_no   : {r.read_no}")
    print(f"stat0     : 0x{r.stat0:02X}")
    print(f"stat1     : 0x{r.stat1:02X}")


def append_csv_row(path: Path, r: LiveReading) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iso_datetime",
                "year",
                "month",
                "day",
                "weekday",
                "hour",
                "minute",
                "second",
                "value",
                "raw_value",
                "lux_value",
                "display_counts",
                "raw_counts",
                "unit",
                "range_name",
                "mode",
                "hold",
                "apo",
                "power",
                "view",
                "memstat",
                "mem_no",
                "read_no",
                "stat0",
                "stat1",
            ],
        )
        if not exists:
            writer.writeheader()
        row = asdict(r)
        row["iso_datetime"] = r.iso_datetime
        row["lux_value"] = r.lux_value
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM6")
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV log file")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("read")
    sub.add_parser("raw")
    sub.add_parser("get-lux")
    p_stream = sub.add_parser("stream")
    p_stream.add_argument("--interval", type=float, default=1.0)

    args = parser.parse_args()

    with LT68(args.port, timeout=args.timeout) as meter:
        if args.cmd == "read":
            r = meter.read_live()
            print_human(r)
            if args.csv:
                append_csv_row(args.csv, r)
            return 0

        if args.cmd == "raw":
            raw = meter.read_live_raw()
            print(raw.hex(" "))
            return 0

        if args.cmd == "get-lux":
            r = meter.get_lux_reading()
            print(json.dumps({
                "iso_datetime": r.iso_datetime,
                "lux": r.lux_value,
                "range": r.range_name,
                "overloaded": r.overloaded,
            }))
            if args.csv:
                append_csv_row(args.csv, r)
            return 0

        if args.cmd == "stream":
            while True:
                r = meter.get_lux_reading()
                print(json.dumps({
                    "iso_datetime": r.iso_datetime,
                    "lux": r.lux_value,
                    "range": r.range_name,
                    "overloaded": r.overloaded,
                }))
                if args.csv:
                    append_csv_row(args.csv, r)
                time.sleep(args.interval)

        raise RuntimeError("unknown command")


if __name__ == "__main__":
    sys.exit(main())
