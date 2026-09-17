"""Unit tests for calibration_model.py -- no Qt, no hardware.

Covers docs/lightbox-calibration-plan.md section 8.2, items 1-8.
"""
import math

import pytest

import calibration_model as cm


# ---------------------------------------------------------------------------
# 1. Serpentine ordering
# ---------------------------------------------------------------------------


def test_serpentine_order_77_entries_no_repeats_starts_at_A1():
    grid = cm.GridConfig()
    labels = cm.serpentine_labels(grid)

    assert len(labels) == 77
    assert len(set(labels)) == 77
    assert labels[0] == "A1"


def test_serpentine_row1_left_to_right_row2_right_to_left():
    grid = cm.GridConfig()
    labels = cm.serpentine_labels(grid)

    row1 = labels[0:11]
    row2 = labels[11:22]
    assert row1 == [f"{cm.col_letter(i)}1" for i in range(11)]
    assert row2 == [f"{cm.col_letter(i)}2" for i in range(10, -1, -1)]


# ---------------------------------------------------------------------------
# 2. Label / coordinate round-trip
# ---------------------------------------------------------------------------


def test_label_coordinate_round_trip():
    col_index, row_index = cm.parse_node_label("F4")
    assert (col_index, row_index) == (5, 3)

    x_mm, y_mm = cm.node_coords_mm(col_index, row_index, pitch_mm=50)
    assert (x_mm, y_mm) == (250, 150)

    assert cm.node_label(col_index, row_index) == "F4"


# ---------------------------------------------------------------------------
# 3. Panel footprint selection
# ---------------------------------------------------------------------------


def test_panel_footprint_matches_expected_node_set():
    grid = cm.GridConfig()
    # 200mm x 150mm panel centred on an 11x7 @ 50mm grid (centre = F4 at 250,150)
    # -> covers cols D..H (200/2=100mm each side of x=250 -> 150..350, inclusive
    # of the 100mm boundary) and rows 3..5 (150/2=75mm each side of y=150 ->
    # 75..225 -> rows y=100,150,200).
    covered = cm.covered_nodes(grid, panel_width_mm=200, panel_height_mm=150)
    expected = {
        f"{col}{row}"
        for col in "DEFGH"
        for row in (3, 4, 5)
    }
    assert set(covered) == expected
    assert cm.footprint_warning(covered) is None


def test_panel_footprint_warns_when_fewer_than_4_nodes():
    grid = cm.GridConfig()
    covered = cm.covered_nodes(grid, panel_width_mm=10, panel_height_mm=10)
    assert len(covered) < 4
    warning = cm.footprint_warning(covered)
    assert warning is not None
    assert "smaller than" in warning


# ---------------------------------------------------------------------------
# 4. Drift normalisation
# ---------------------------------------------------------------------------


def test_drift_normalisation_recovers_flat_field():
    ref_start, ref_end = 100.0, 105.0  # 5% linear drift
    t_start, t_end = 0.0, 100.0
    flat_value = 42.0

    for t_i in (0.0, 25.0, 50.0, 75.0, 100.0):
        ref_interp_t = cm.interpolate_reference(ref_start, ref_end, t_start, t_end, t_i)
        measured = flat_value * ref_interp_t / ref_start  # simulate drifted instrument
        normalised = cm.normalize_value(measured, ref_start, ref_interp_t)
        assert normalised == pytest.approx(flat_value, rel=0.001)

    assert cm.drift_pct(ref_start, ref_end) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 5. K computation
# ---------------------------------------------------------------------------


def test_K_computation_hand_worked_example():
    values_by_node = {"E3": 900.0, "F3": 950.0, "G3": 900.0, "E4": 950.0, "F4": 1000.0,
                       "G4": 950.0, "E5": 900.0, "F5": 950.0, "G5": 900.0}
    covered = ["E3", "F3", "G3", "E4", "F4", "G4", "E5", "F5", "G5"]
    sensor_node = "B6"
    values_by_node[sensor_node] = 800.0

    pm = cm.panel_mean(values_by_node, covered)
    assert pm == pytest.approx(933.333, rel=1e-3)

    k = cm.compute_K(pm, values_by_node[sensor_node])
    assert k == pytest.approx(933.333 / 800.0, rel=1e-3)

    stats = cm.uniformity_stats(values_by_node, covered)
    assert stats["panel_min"] == 900.0
    assert stats["panel_max"] == 1000.0
    assert stats["uniformity_pct"] == pytest.approx((1000 - 900) / (1000 + 900) * 100.0)


# ---------------------------------------------------------------------------
# 6. Level independence
# ---------------------------------------------------------------------------


def test_level_independence_constant_shape_field():
    # Same spatial shape at 3 levels -> K should be ~constant.
    k_by_level = {1: 1.1667, 2: 1.1668, 3: 1.1666}
    report = cm.level_independence_report(k_by_level)
    assert report["k_spread_pct"] < 0.1
    assert report["single_factor_sufficient"] is True


def test_level_independence_flags_real_spread():
    k_by_level = {1: 1.0, 2: 1.2, 3: 1.5}
    report = cm.level_independence_report(k_by_level, variac_by_level={1: 20, 2: 60, 3: 110})
    assert report["k_spread_pct"] > 2.0
    assert report["single_factor_sufficient"] is False


# ---------------------------------------------------------------------------
# 7. CSV round-trip
# ---------------------------------------------------------------------------


def test_captures_csv_round_trip_and_supersede(tmp_path):
    csv_path = tmp_path / "captures.csv"
    grid = cm.GridConfig()

    record1 = cm.make_capture_record(
        campaign_id="cal-test", level_index=1, variac_vac=42.0, quantity="lux",
        capture_kind="grid", node_label_="D4", grid=grid, sequence_index=30,
        samples=[100.0, 101.0, 99.5], duration_s=3.0, sensor_port="COM6",
        timestamp_iso="2026-09-16T14:22:00",
    )
    cm.append_capture(csv_path, record1)

    # A redo of the same node: new row, old one gets superseded.
    record2 = cm.make_capture_record(
        campaign_id="cal-test", level_index=1, variac_vac=42.0, quantity="lux",
        capture_kind="grid", node_label_="D4", grid=grid, sequence_index=30,
        samples=[105.0, 104.0, 106.0], duration_s=3.0, sensor_port="COM6",
        timestamp_iso="2026-09-16T14:25:00",
    )
    cm.mark_superseded(csv_path, record1["capture_id"])
    cm.append_capture(csv_path, record2)

    rows = cm.read_captures(csv_path)
    assert len(rows) == 2

    active = cm.active_captures(rows)
    assert len(active) == 1
    assert active[0]["capture_id"] == record2["capture_id"]
    assert active[0]["mean"] == pytest.approx(105.0, rel=1e-6)
    assert active[0]["node_label"] == "D4"
    assert active[0]["col_index"] == 3
    assert active[0]["row_index"] == 3

    # Undo: restoring the original row (and re-superseding the redo) recovers
    # the exact original data -- nothing was ever actually deleted.
    cm.mark_superseded(csv_path, record1["capture_id"], superseded=False)
    cm.mark_superseded(csv_path, record2["capture_id"], superseded=True)
    restored = cm.active_captures(cm.read_captures(csv_path))
    assert len(restored) == 1
    assert restored[0]["capture_id"] == record1["capture_id"]
    assert restored[0]["mean"] == pytest.approx(sum([100.0, 101.0, 99.5]) / 3, rel=1e-6)


# ---------------------------------------------------------------------------
# 8. Resume
# ---------------------------------------------------------------------------


def test_resume_cursor_restores_mid_pass(tmp_path):
    csv_path = tmp_path / "captures.csv"
    grid = cm.GridConfig()
    order = cm.serpentine_labels(grid)

    # ref_start, then the first 30 grid nodes of the lux pass.
    cm.append_capture(csv_path, cm.make_capture_record(
        campaign_id="cal-test", level_index=1, variac_vac=42.0, quantity="lux",
        capture_kind="ref_start", node_label_="F4", grid=grid, sequence_index=-1,
        samples=[500.0] * 3, duration_s=3.0, sensor_port="COM6",
    ))
    for idx in range(30):
        cm.append_capture(csv_path, cm.make_capture_record(
            campaign_id="cal-test", level_index=1, variac_vac=42.0, quantity="lux",
            capture_kind="grid", node_label_=order[idx], grid=grid, sequence_index=idx,
            samples=[500.0] * 3, duration_s=3.0, sensor_port="COM6",
        ))

    captures = cm.read_captures(csv_path)
    cursor = cm.next_cursor(captures, grid, reference_node="F4")

    assert cursor["level_index"] == 1
    assert cursor["pass"] == "lux"
    assert cursor["next_kind"] == "grid"
    assert cursor["sequence_index"] == 30
    assert cursor["node_label"] == order[30]


def test_resume_cursor_moves_to_ref_end_then_next_pass(tmp_path):
    csv_path = tmp_path / "captures.csv"
    grid = cm.GridConfig()
    order = cm.serpentine_labels(grid)

    cm.append_capture(csv_path, cm.make_capture_record(
        campaign_id="cal-test", level_index=1, variac_vac=42.0, quantity="lux",
        capture_kind="ref_start", node_label_="F4", grid=grid, sequence_index=-1,
        samples=[500.0] * 3, duration_s=3.0, sensor_port="COM6",
    ))
    for idx, label in enumerate(order):
        cm.append_capture(csv_path, cm.make_capture_record(
            campaign_id="cal-test", level_index=1, variac_vac=42.0, quantity="lux",
            capture_kind="grid", node_label_=label, grid=grid, sequence_index=idx,
            samples=[500.0] * 3, duration_s=3.0, sensor_port="COM6",
        ))

    captures = cm.read_captures(csv_path)
    cursor = cm.next_cursor(captures, grid, reference_node="F4")
    assert cursor["next_kind"] == "ref_end"
    assert cursor["pass"] == "lux"

    cm.append_capture(csv_path, cm.make_capture_record(
        campaign_id="cal-test", level_index=1, variac_vac=42.0, quantity="lux",
        capture_kind="ref_end", node_label_="F4", grid=grid, sequence_index=-1,
        samples=[510.0] * 3, duration_s=3.0, sensor_port="COM6",
    ))
    captures = cm.read_captures(csv_path)
    cursor = cm.next_cursor(captures, grid, reference_node="F4")
    assert cursor["pass"] == "irr"
    assert cursor["next_kind"] == "ref_start"


def test_resume_cursor_none_once_level_fully_complete(tmp_path):
    csv_path = tmp_path / "captures.csv"
    grid = cm.GridConfig()
    order = cm.serpentine_labels(grid)

    for quantity in ("lux", "irr"):
        cm.append_capture(csv_path, cm.make_capture_record(
            campaign_id="cal-test", level_index=1, variac_vac=42.0, quantity=quantity,
            capture_kind="ref_start", node_label_="F4", grid=grid, sequence_index=-1,
            samples=[500.0] * 3, duration_s=3.0, sensor_port="COM6",
        ))
        for idx, label in enumerate(order):
            cm.append_capture(csv_path, cm.make_capture_record(
                campaign_id="cal-test", level_index=1, variac_vac=42.0, quantity=quantity,
                capture_kind="grid", node_label_=label, grid=grid, sequence_index=idx,
                samples=[500.0] * 3, duration_s=3.0, sensor_port="COM6",
            ))
        cm.append_capture(csv_path, cm.make_capture_record(
            campaign_id="cal-test", level_index=1, variac_vac=42.0, quantity=quantity,
            capture_kind="ref_end", node_label_="F4", grid=grid, sequence_index=-1,
            samples=[510.0] * 3, duration_s=3.0, sensor_port="COM6",
        ))

    captures = cm.read_captures(csv_path)
    assert cm.next_cursor(captures, grid, reference_node="F4") is None


# ---------------------------------------------------------------------------
# Extra coverage: stats and campaign.json round-trip
# ---------------------------------------------------------------------------


def test_compute_stats_basic():
    stats = cm.compute_stats([10.0, 10.0, 10.0])
    assert stats["n"] == 3
    assert stats["mean"] == 10.0
    assert stats["stdev"] == 0.0
    assert stats["three_sigma_pct"] == 0.0


def test_compute_stats_single_sample_has_no_stdev():
    stats = cm.compute_stats([10.0])
    assert stats["n"] == 1
    assert stats["stdev"] is None
    assert stats["three_sigma_pct"] is None


def test_analyze_campaign_end_to_end(tmp_path):
    csv_path = tmp_path / "captures.csv"
    grid = cm.GridConfig()
    order = cm.serpentine_labels(grid)
    covered = cm.covered_nodes(grid, panel_width_mm=200, panel_height_mm=150)

    def write_pass(level_index, variac, quantity, base):
        cm.append_capture(csv_path, cm.make_capture_record(
            campaign_id="cal-test", level_index=level_index, variac_vac=variac, quantity=quantity,
            capture_kind="ref_start", node_label_="F4", grid=grid, sequence_index=-1,
            samples=[base] * 3, duration_s=3.0, sensor_port="COM6", timestamp_iso=0,
        ))
        for idx, label in enumerate(order):
            col, row = cm.parse_node_label(label)
            value = base * (1.2 if label in covered else 1.0)
            cm.append_capture(csv_path, cm.make_capture_record(
                campaign_id="cal-test", level_index=level_index, variac_vac=variac, quantity=quantity,
                capture_kind="grid", node_label_=label, grid=grid, sequence_index=idx,
                samples=[value] * 3, duration_s=3.0, sensor_port="COM6", timestamp_iso=float(idx + 1),
            ))
        cm.append_capture(csv_path, cm.make_capture_record(
            campaign_id="cal-test", level_index=level_index, variac_vac=variac, quantity=quantity,
            capture_kind="ref_end", node_label_="F4", grid=grid, sequence_index=-1,
            samples=[base] * 3, duration_s=3.0, sensor_port="COM6", timestamp_iso=float(len(order) + 1),
        ))

    for level_index, variac in ((1, 20.0), (2, 60.0), (3, 110.0)):
        write_pass(level_index, variac, "lux", base=100.0)
        write_pass(level_index, variac, "irr", base=50.0)

    config = cm.default_campaign_config(
        campaign_id="cal-test", operator="AK", grid=grid, reference_node="F4",
        panel={"width_mm": 200, "height_mm": 150}, sensor_node="B6",
    )
    captures = cm.read_captures(csv_path)
    analysis = cm.analyze_campaign(captures, config)

    assert set(analysis["levels"]) == {1, 2, 3}
    for level_index in (1, 2, 3):
        lux = analysis["levels"][level_index]["lux"]
        assert lux["drift_pct"] == pytest.approx(0.0)
        assert lux["K"] == pytest.approx(1.2, rel=1e-6)

    report = analysis["independence"]["lux"]
    assert report["single_factor_sufficient"] is True
    assert len(analysis["correction_rows"]) == 6
    assert 1 in analysis["ratio_maps"]
    assert analysis["ratio_maps"][1]["F4"] == pytest.approx(2.0, rel=1e-6)


def test_campaign_config_round_trip(tmp_path):
    campaign_id = cm.new_campaign_id("lightbox", when=1758000000)
    assert campaign_id.startswith("cal-")

    config = cm.default_campaign_config(campaign_id=campaign_id, operator="AK")
    cm.save_campaign_config(tmp_path, config)

    loaded = cm.load_campaign_config(tmp_path)
    assert loaded["campaign_id"] == campaign_id
    assert loaded["operator"] == "AK"
    assert cm.grid_from_campaign(loaded) == cm.GridConfig()
