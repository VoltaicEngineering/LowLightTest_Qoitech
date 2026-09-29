"""Lightbox-mode maths and file handling -- no Qt event loop, no hardware.

Most tests run on a small synthetic campaign (an exact power law in VAC times
a fixed radial light pattern) so the expected numbers are known exactly. A few
extra checks run against the real 2026-09-17 campaign when it is present
(calibration/ is not in git)."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

import calibration_model as cm
import calibration_report as cr
import norm_analysis
from IV_curve_CURRENT_V3 import SUMMARY_CSV_FIELDNAMES, append_summary_csv

VACS = [20.0, 30.0, 45.0, 60.0]
N_IRR, N_LUX = 4.0, 3.0
REAL_CAMPAIGN = Path(__file__).resolve().parents[1] / "calibration" / "cal-20260917-113349-lightbox"


def shape(c, r, grid):
    x, y = c * grid.pitch_mm, r * grid.pitch_mm
    xc, yc = cr.grid_centre(grid)
    return 1.0 - 0.5 * (((x - xc) / (xc + 50)) ** 2 + ((y - yc) / (yc + 50)) ** 2)


def true_value(q, vac, c, r, grid):
    base = 100.0 * (vac / 60.0) ** N_IRR if q == "irr" else 3000.0 * (vac / 60.0) ** N_LUX
    return base * shape(c, r, grid)


def make_campaign(folder: Path, bad=None, irr_offset=cr.IRR_ZERO_OFFSET_WM2):
    """Write campaign.json + captures.csv. bad = {(vac, q): {label: factor}}
    multiplies those nodes' readings (a fake shadow)."""
    grid = cm.GridConfig()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "campaign.json").write_text(json.dumps({
        "campaign_id": "cal-test", "grid": {"cols": grid.cols, "rows": grid.rows, "pitch_mm": grid.pitch_mm},
    }), encoding="utf-8")
    seq = 0
    for li, vac in enumerate(VACS, start=1):
        for q in ("lux", "irr"):
            off = irr_offset if q == "irr" else 0.0
            ref = true_value(q, vac, 5, 3, grid) + off

            def rec(kind, label, value):
                nonlocal seq
                seq += 1
                cm.append_capture(cm.captures_csv_path(folder), cm.make_capture_record(
                    campaign_id="cal-test", level_index=li, variac_vac=vac, quantity=q, capture_kind=kind,
                    node_label_=label, grid=grid, sequence_index=seq, samples=[value] * 5, duration_s=0.5,
                    sensor_port="COMX", timestamp_iso=f"2026-09-17T12:{seq // 60:02d}:{seq % 60:02d}",
                ))
            rec("ref_start", "F4", ref)
            for c, r in cm.serpentine_indices(grid):
                label = cm.node_label(c, r)
                factor = (bad or {}).get((vac, q), {}).get(label, 1.0)
                rec("grid", label, true_value(q, vac, c, r, grid) * factor + off)
            rec("ref_end", "F4", ref)
    return grid


@pytest.fixture
def campaign(tmp_path):
    folder = tmp_path / "cal-test"
    grid = make_campaign(folder)
    return folder, grid


def test_expected_at_vac_exact_at_measured_level(campaign):
    folder, grid = campaign
    _, grid, passes = cr.load_calibration(folder)
    exp = cr.expected_at_vac(passes, grid, 30.0, {"irr": "B6", "lux": "J6"})
    c, r = cm.parse_node_label("B6")
    assert exp["nodes"]["irr"]["value"] == pytest.approx(true_value("irr", 30.0, c, r, grid), rel=1e-6)
    c, r = cm.parse_node_label("J6")
    assert exp["nodes"]["lux"]["value"] == pytest.approx(true_value("lux", 30.0, c, r, grid), rel=1e-6)


def test_expected_at_vac_between_levels_follows_power_law(campaign):
    folder, _ = campaign
    _, grid, passes = cr.load_calibration(folder)
    exp = cr.expected_at_vac(passes, grid, 37.0, {"irr": "A1", "lux": "K7"}, 200, 150)
    assert exp["nodes"]["irr"]["value"] == pytest.approx(true_value("irr", 37.0, 0, 0, grid), rel=1e-6)
    assert exp["nodes"]["irr"]["n"] == pytest.approx(N_IRR, rel=1e-6)
    assert exp["nodes"]["lux"]["n"] == pytest.approx(N_LUX, rel=1e-6)
    p30 = cr.panel_average(passes[(2, "irr")]["filled"], grid, 200, 150)
    assert exp["irr_panel_avg"] == pytest.approx(p30 * (37 / 30) ** N_IRR, rel=1e-6)


def test_expected_at_vac_out_of_range_raises(campaign):
    folder, _ = campaign
    _, grid, passes = cr.load_calibration(folder)
    with pytest.raises(ValueError, match="outside the calibrated range"):
        cr.expected_at_vac(passes, grid, 15.0, {"irr": "F4"})
    with pytest.raises(ValueError):
        cr.expected_at_vac(passes, grid, 30.0, {"irr": "Z9"})


def test_vac_for_target_round_trip_and_sensor_target(campaign):
    folder, _ = campaign
    _, grid, passes = cr.load_calibration(folder)
    setpoint = 50.0
    res = cr.vac_for_target(passes, grid, 200, 150, setpoint, "irr", {"irr": "B6"})
    exp = cr.expected_at_vac(passes, grid, res["vac"], {"irr": "B6"}, 200, 150)
    assert exp["irr_panel_avg"] == pytest.approx(setpoint, rel=1e-6)
    assert res["sensors"]["irr"]["value"] == pytest.approx(exp["nodes"]["irr"]["value"], rel=1e-9)
    assert res["n"] == pytest.approx(N_IRR, rel=1e-6)


def test_vac_for_target_below_range_says_lowest_reachable(campaign):
    folder, _ = campaign
    _, grid, passes = cr.load_calibration(folder)
    with pytest.raises(ValueError, match="Lowest reachable for this size: .* at 20 VAC. Add lower calibration levels"):
        cr.vac_for_target(passes, grid, 200, 150, 0.5, "irr")


def test_exclusions_mask_block_and_fill_from_adjacent_levels(tmp_path):
    folder = tmp_path / "cal-shadow"
    block = ["I5", "J5", "K5", "I6", "J6", "K6", "I7", "J7", "K7"]
    grid = make_campaign(folder, bad={(30.0, "irr"): {n: 0.5 for n in block}})
    (folder / cr.EXCLUSIONS_FILE).write_text(
        "vac,quantity,nodes,reason\n30,irr," + " ".join(block) + ",test shadow\n", encoding="utf-8")
    _, grid, passes = cr.load_calibration(folder)
    p = passes[(2, "irr")]
    flagged = {cm.node_label(c, r) for r, c in np.argwhere(p["outliers"])}
    assert flagged == set(block)
    # The middle of the block has no good neighbour; the adjacent levels'
    # pattern puts it back on the true value (exact for this synthetic data).
    c, r = cm.parse_node_label("K6")
    assert p["filled"][r, c] == pytest.approx(true_value("irr", 30.0, c, r, grid), rel=1e-6)
    # Other passes untouched.
    assert not passes[(2, "lux")]["outliers"].any()


def test_exclusions_bad_node_rejected(tmp_path):
    folder = tmp_path / "cal-bad"
    make_campaign(folder)
    (folder / cr.EXCLUSIONS_FILE).write_text("vac,quantity,nodes,reason\n30,irr,Z12,oops\n", encoding="utf-8")
    with pytest.raises(ValueError):
        cr.load_calibration(folder)


def test_panel_config_width_height_round_trip_and_old_files(tmp_path):
    cfg = {"P124N042": {"cell_type": "X", "cells_series": 6, "cells_parallel": 1, "width_mm": 200.0, "height_mm": 150.0},
           "P1": {"cell_type": "", "cells_series": 1, "cells_parallel": 1, "width_mm": None, "height_mm": None}}
    path = tmp_path / "panel_config.csv"
    norm_analysis.save_panel_config(path, cfg)
    assert norm_analysis.load_panel_config(path) == cfg
    old = tmp_path / "old.csv"
    old.write_text("panel_name,cell_type,cells_series,cells_parallel\nP9,X,2,1\n", encoding="utf-8")
    loaded = norm_analysis.load_panel_config(old)
    assert loaded["P9"]["width_mm"] is None and loaded["P9"]["height_mm"] is None
    assert norm_analysis.lookup_panel("p124n042 ", cfg)[0] == "P124N042"
    with pytest.raises(ValueError):
        norm_analysis.parse_dimension_mm("-5")


def test_summary_csv_old_header_is_upgraded_not_misaligned(tmp_path):
    path = tmp_path / "LowLightTesting_summary.csv"
    old_fields = SUMMARY_CSV_FIELDNAMES[: SUMMARY_CSV_FIELDNAMES.index("csv_path") + 1]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=old_fields)
        w.writeheader()
        w.writerow({k: "" for k in old_fields} | {"panel_name": "OLD", "Wp": "1.5", "csv_path": "old.csv"})
    append_summary_csv(path, {"panel_name": "NEW", "Wp": 2.5, "vac_set": 53.5, "irr_node": "B6"})
    with path.open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == SUMMARY_CSV_FIELDNAMES
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["panel_name"] for r in rows] == ["OLD", "NEW"]
    assert rows[0]["csv_path"] == "old.csv" and rows[0]["vac_set"] == ""
    assert rows[1]["vac_set"] == "53.5" and rows[1]["irr_node"] == "B6"


# ---------------------------------------------------------------------------
# Real 2026-09-17 campaign (skipped when calibration/ isn't on this machine)
# ---------------------------------------------------------------------------

real = pytest.mark.skipif(not REAL_CAMPAIGN.exists(), reason="calibration/cal-20260917-113349-lightbox not present")


@real
def test_real_campaign_matches_report_calculator():
    _, grid, passes = cr.load_calibration(REAL_CAMPAIGN)
    res = cr.vac_for_target(passes, grid, 200, 150, 100.0, "irr", {"irr": "B6", "lux": "J6"})
    assert res["vac"] == pytest.approx(53.3, abs=0.05)
    assert res["sensors"]["irr"]["value"] == pytest.approx(82.57, rel=1e-3)
    assert res["sensors"]["lux"]["value"] == pytest.approx(965.4, rel=1e-3)


@real
def test_real_campaign_29vac_shadow_excluded_and_filled():
    if not (REAL_CAMPAIGN / cr.EXCLUSIONS_FILE).exists():
        pytest.skip("no exclusions.csv in the real campaign")
    _, grid, passes = cr.load_calibration(REAL_CAMPAIGN)
    level = next(l for (l, q), p in passes.items() if q == "irr" and abs(p["vac"] - 29) < 0.3)
    p = passes[(level, "irr")]
    c, r = cm.parse_node_label("K6")
    assert p["outliers"][r, c]
    # K6 relative to F4 should sit near the adjacent levels' pattern.
    neighbours = [lv for lv in passes if lv[1] == "irr" and abs(passes[lv]["vac"] - 29) in (4.0, 8.0)]
    ratios = [passes[lv]["filled"][r, c] / passes[lv]["filled"][3, 5] for lv in neighbours]
    assert p["filled"][r, c] / p["filled"][3, 5] == pytest.approx(math.exp(np.mean(np.log(ratios))), rel=0.05)
