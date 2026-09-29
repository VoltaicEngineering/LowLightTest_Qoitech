"""harvest()/short_circuit() against a fake Otii -- no hardware.

Covers the 2026-09 slowdown/stability fixes: the live preview fetches only new
samples, the Arc is always switched off after a timeout or an Otii error, and
short_circuit() can delete its recording."""
import time

import pytest

import IV_curve_CURRENT_V3 as iv


@pytest.fixture(autouse=True)
def fast_pacing(monkeypatch):
    monkeypatch.setitem(iv.SETTING, "interval", 0.002)
    monkeypatch.setattr(iv, "ADAPTIVE_INTERVAL_MIN_S", 0.001)


class FakeRecording:
    def __init__(self, project):
        self.id = 7
        self.project = project
        self.fetched = 0
        self.deleted = False

    def get_channel_data_count(self, device_id, channel):
        return int((time.monotonic() - self.project.t0) * 1000)  # 1 kHz

    def get_channel_data(self, device_id, channel, index, count):
        self.fetched += count
        return {"values": [0.0] * count}

    def delete(self):
        self.deleted = True
        self.id = -1


class FakeProject:
    def __init__(self):
        self.recording = None
        self.running = False

    def start_recording(self):
        self.t0 = time.monotonic()
        self.running = True
        self.recording = FakeRecording(self)

    def stop_recording(self):
        self.running = False

    def get_last_recording(self):
        return self.recording


class FakeArc:
    id = "arc1"

    def __init__(self, cutoff_steps=None, fail_after=None):
        self.main = False
        self.mv_reads = 0
        self.cutoff_steps = cutoff_steps
        self.fail_after = fail_after

    def set_main_current(self, value):
        if self.fail_after is not None and value != 0 and self.mv_reads >= self.fail_after:
            raise RuntimeError("Transaction id mismatch")

    def set_power_regulation(self, mode): pass
    def enable_channel(self, channel, enable): pass
    def set_channel_samplerate(self, channel, rate): pass

    def set_main(self, on):
        self.main = on

    def get_value(self, channel):
        if channel == "mc":
            return -0.001
        self.mv_reads += 1
        time.sleep(0.005)
        return 0.1 if self.cutoff_steps and self.mv_reads >= self.cutoff_steps else 1.0


def test_live_preview_fetches_only_new_samples():
    project, arc = FakeProject(), FakeArc(cutoff_steps=120)
    sizes, stats = [], {}
    rec = iv.harvest(project, [arc], 10.0, timeout_seconds=30, stats=stats,
                     live_data_cb=lambda mv, mc: sizes.append(len(mv)), live_data_interval_s=0.1)
    assert len(sizes) >= 3 and sizes == sorted(sizes)
    # Each sample of mv and mc fetched once, not once per refresh.
    assert rec.fetched <= 2 * sizes[-1] + 10
    assert stats["steps"] == 120 and stats["duration_s"] > 0
    assert not arc.main and not project.running


def test_timeout_switches_arc_off():
    project, arc = FakeProject(), FakeArc()
    with pytest.raises(TimeoutError):
        iv.harvest(project, [arc], 10.0, timeout_seconds=0.2)
    assert not arc.main and not project.running and not iv._arcs_active


def test_otii_error_mid_sweep_switches_arc_off():
    project, arc = FakeProject(), FakeArc(fail_after=5)
    with pytest.raises(RuntimeError, match="mismatch"):
        iv.harvest(project, [arc], 10.0, timeout_seconds=5)
    assert not arc.main and not project.running


def test_short_circuit_deletes_recording_and_winds_down_on_failure():
    project, arc = FakeProject(), FakeArc()
    assert iv.short_circuit(project, [arc], delete_recording_after=True) == pytest.approx(0.001)
    assert project.recording.deleted and arc.main  # left on for the sweep that follows

    class NoReading(FakeArc):
        def get_value(self, channel):
            raise RuntimeError("no reply")

    project, arc = FakeProject(), NoReading()
    with pytest.raises(RuntimeError):
        iv.short_circuit(project, [arc])
    assert not arc.main and not project.running
