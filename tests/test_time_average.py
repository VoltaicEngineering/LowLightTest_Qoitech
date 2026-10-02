"""Time-average app: the lux poller's pause is configurable (the app reads
back-to-back) and offset text parsing treats bad input as 'no offset'."""
import inspect

import sensors
from time_average import parse_offset


def test_lux_poll_loop_interval_defaults_to_existing_cadence():
    param = inspect.signature(sensors.lux_poll_loop).parameters["interval_s"]
    assert param.default == sensors.LUX_POLL_INTERVAL_S


def test_parse_offset():
    assert parse_offset("0.22") == 0.22
    assert parse_offset(" -1.5 ") == -1.5
    assert parse_offset("") is None
    assert parse_offset("-") is None
    assert parse_offset("nan") is None
