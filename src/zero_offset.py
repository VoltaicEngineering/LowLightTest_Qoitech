"""Irradiance sensor zero-offset capture (the reading at zero irradiance).

Samples are copied out of the live irradiance history as they arrive, so a
capture can run longer than the history buffer keeps. No Qt here; the tab in
low_light_app.py drives it from a timer."""
from __future__ import annotations

import math
import time


class ZeroOffsetCapture:
    def __init__(self, history):
        self.history = history  # sensors.TimeSeriesBuffer
        self.times: list[float] = []
        self.values: list[float] = []
        self.started_at: float | None = None
        self.stopped_at: float | None = None
        self._last_ts: float | None = None

    @property
    def running(self) -> bool:
        return self.started_at is not None and self.stopped_at is None

    def start(self, now: float | None = None) -> None:
        self.times, self.values = [], []
        self.started_at = time.time() if now is None else now
        self.stopped_at = None
        self._last_ts = self.started_at

    def poll(self) -> int:
        """Copy samples that arrived since the last poll. Returns how many."""
        if not self.running:
            return 0
        times, values = self.history.snapshot()
        added = 0
        for t, v in zip(times, values):
            if t > self._last_ts and v is not None and math.isfinite(v):
                self.times.append(t)
                self.values.append(float(v))
                added += 1
        if times:
            self._last_ts = max(self._last_ts, times[-1])
        return added

    def stop(self, now: float | None = None) -> None:
        if self.running:
            self.poll()
            self.stopped_at = time.time() if now is None else now

    @property
    def n(self) -> int:
        return len(self.values)

    def mean(self) -> float | None:
        return sum(self.values) / len(self.values) if self.values else None

    def stdev(self) -> float | None:
        if len(self.values) < 2:
            return None
        m = self.mean()
        return math.sqrt(sum((v - m) ** 2 for v in self.values) / (len(self.values) - 1))

    def running_mean(self) -> list[float]:
        """Cumulative average after each sample."""
        out, total = [], 0.0
        for i, v in enumerate(self.values, start=1):
            total += v
            out.append(total / i)
        return out

    def elapsed(self, now: float | None = None) -> float:
        if self.started_at is None:
            return 0.0
        end = self.stopped_at if self.stopped_at is not None else (time.time() if now is None else now)
        return end - self.started_at
