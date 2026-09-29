"""Irradiance zero-offset capture: only samples taken between Start and Stop
count, each once, and the running average ends on the overall mean."""
import pytest

from sensors import TimeSeriesBuffer
from zero_offset import ZeroOffsetCapture


def test_capture_uses_only_samples_between_start_and_stop():
    hist = TimeSeriesBuffer(max_age_s=1e9)
    hist.add(5.0, ts=99.0)  # before Start: ignored
    cap = ZeroOffsetCapture(hist)
    cap.start(now=100.0)
    for i, v in enumerate([0.20, 0.24, 0.22]):
        hist.add(v, ts=100.5 + i)
    assert cap.poll() == 3
    assert cap.poll() == 0  # nothing new: no double counting
    hist.add(0.26, ts=104.0)
    hist.add(float("nan"), ts=104.5)  # dropped
    cap.stop(now=105.0)
    hist.add(9.0, ts=106.0)  # after Stop: ignored
    cap.poll()
    assert cap.values == [0.20, 0.24, 0.22, 0.26]
    assert cap.mean() == pytest.approx(0.23)
    assert cap.running_mean() == pytest.approx([0.20, 0.22, 0.22, 0.23])
    assert cap.stdev() == pytest.approx(0.0258199, rel=1e-5)
    assert cap.elapsed() == pytest.approx(5.0)
    assert not cap.running


def test_capture_outlives_the_history_buffer():
    # The history only keeps 10 s; the capture keeps everything it copied.
    hist = TimeSeriesBuffer(max_age_s=10)
    cap = ZeroOffsetCapture(hist)
    cap.start(now=0.0)
    for t in range(1, 60):
        hist.add(0.2, ts=float(t))
        cap.poll()
    cap.stop(now=60.0)
    assert cap.n == 59 and cap.mean() == pytest.approx(0.2)


def test_restart_clears_previous_capture():
    hist = TimeSeriesBuffer(max_age_s=1e9)
    cap = ZeroOffsetCapture(hist)
    cap.start(now=0.0)
    hist.add(1.0, ts=1.0)
    cap.stop(now=2.0)
    cap.start(now=3.0)
    hist.add(0.5, ts=4.0)
    cap.poll()
    assert cap.values == [0.5]
