"""Pure logic for the lightbox spatial calibration tool.

Grid geometry, serpentine ordering, capture statistics, drift normalisation,
correction-factor maths, and CSV/JSON I/O for a calibration campaign. Zero Qt
imports, zero hardware I/O -- see docs/lightbox-calibration-plan.md sections
5 and 6. This module is unit-tested headlessly in tests/test_calibration_model.py.
"""
from __future__ import annotations

import csv
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Grid geometry (plan section 2)
# ---------------------------------------------------------------------------

DEFAULT_COLS = 11
DEFAULT_ROWS = 7
DEFAULT_PITCH_MM = 50.0
DEFAULT_REFERENCE_NODE = "F4"
DEFAULT_DWELL_SECONDS = 3.0
DEFAULT_WARMUP_SECONDS = 120.0
DEFAULT_DRIFT_FLAG_PCT = 3.0
MIN_COVERED_NODES_WARNING = 4
K_SPREAD_SINGLE_FACTOR_THRESHOLD_PCT = 2.0
MIN_LEVELS_FOR_LINEAR_FIT = 4


@dataclass(frozen=True)
class GridConfig:
    cols: int = DEFAULT_COLS
    rows: int = DEFAULT_ROWS
    pitch_mm: float = DEFAULT_PITCH_MM

    def __post_init__(self):
        if not (1 <= self.cols <= 26):
            raise ValueError("cols must be between 1 and 26 (single-letter column labels)")
        if self.rows < 1:
            raise ValueError("rows must be >= 1")
        if self.pitch_mm <= 0:
            raise ValueError("pitch_mm must be > 0")

    @property
    def node_count(self) -> int:
        return self.cols * self.rows


def col_letter(col_index: int) -> str:
    if not (0 <= col_index < 26):
        raise ValueError(f"col_index out of range: {col_index}")
    return chr(ord("A") + col_index)


def col_index_from_letter(letter: str) -> int:
    letter = letter.strip().upper()
    if len(letter) != 1 or not letter.isalpha():
        raise ValueError(f"invalid column letter: {letter!r}")
    return ord(letter) - ord("A")


def node_label(col_index: int, row_index: int) -> str:
    return f"{col_letter(col_index)}{row_index + 1}"


def parse_node_label(label: str) -> tuple[int, int]:
    """'F4' -> (5, 3) (0-based column, 0-based row)."""
    label = label.strip().upper()
    if len(label) < 2:
        raise ValueError(f"invalid node label: {label!r}")
    col_index = col_index_from_letter(label[0])
    try:
        row_index = int(label[1:]) - 1
    except ValueError:
        raise ValueError(f"invalid node label: {label!r}") from None
    if row_index < 0:
        raise ValueError(f"invalid node label: {label!r}")
    return col_index, row_index


def node_coords_mm(col_index: int, row_index: int, pitch_mm: float) -> tuple[float, float]:
    return col_index * pitch_mm, row_index * pitch_mm


def label_to_coords_mm(label: str, pitch_mm: float) -> tuple[float, float]:
    col_index, row_index = parse_node_label(label)
    return node_coords_mm(col_index, row_index, pitch_mm)


def all_node_labels(grid: GridConfig) -> list[str]:
    return [node_label(c, r) for r in range(grid.rows) for c in range(grid.cols)]


# ---------------------------------------------------------------------------
# Serpentine order (plan section 3): left-to-right on row 1, right-to-left on
# row 2, and so on, so the operator never walks the jig back across the plate.
# ---------------------------------------------------------------------------


def serpentine_indices(grid: GridConfig) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for row_index in range(grid.rows):
        cols_range = range(grid.cols) if row_index % 2 == 0 else range(grid.cols - 1, -1, -1)
        for col_index in cols_range:
            order.append((col_index, row_index))
    return order


def serpentine_labels(grid: GridConfig) -> list[str]:
    return [node_label(c, r) for c, r in serpentine_indices(grid)]


# ---------------------------------------------------------------------------
# Panel footprint (plan section 6.2)
# ---------------------------------------------------------------------------


def covered_nodes(grid: GridConfig, panel_width_mm: float, panel_height_mm: float) -> list[str]:
    """Node labels whose centre falls under a panel of the given size,
    centred on the grid."""
    x_centre = (grid.cols - 1) * grid.pitch_mm / 2.0
    y_centre = (grid.rows - 1) * grid.pitch_mm / 2.0
    covered = []
    for row_index in range(grid.rows):
        for col_index in range(grid.cols):
            x_mm, y_mm = node_coords_mm(col_index, row_index, grid.pitch_mm)
            if abs(x_mm - x_centre) <= panel_width_mm / 2.0 and abs(y_mm - y_centre) <= panel_height_mm / 2.0:
                covered.append(node_label(col_index, row_index))
    return covered


def footprint_warning(covered: list[str]) -> str | None:
    if len(covered) < MIN_COVERED_NODES_WARNING:
        return (
            f"Panel footprint covers only {len(covered)} grid node(s) -- smaller than the "
            f"{MIN_COVERED_NODES_WARNING} needed for the grid pitch to resolve it; the "
            "panel-average will be a poor estimate."
        )
    return None


# ---------------------------------------------------------------------------
# Capture statistics
# ---------------------------------------------------------------------------


def compute_stats(samples: list[float]) -> dict:
    """n, mean, stdev (population, matching low_light_app's np.std convention),
    3*sigma/mean %, min, max. stdev/three_sigma_pct are None when n < 2."""
    n = len(samples)
    if n == 0:
        return {"n": 0, "mean": None, "stdev": None, "three_sigma_pct": None, "min": None, "max": None}

    mean = sum(samples) / n
    lo, hi = min(samples), max(samples)
    if n < 2:
        return {"n": n, "mean": mean, "stdev": None, "three_sigma_pct": None, "min": lo, "max": hi}

    variance = sum((x - mean) ** 2 for x in samples) / n
    stdev = variance ** 0.5
    three_sigma_pct = (3.0 * stdev / mean * 100.0) if mean else None
    return {"n": n, "mean": mean, "stdev": stdev, "three_sigma_pct": three_sigma_pct, "min": lo, "max": hi}


# ---------------------------------------------------------------------------
# captures.csv (plan section 5)
# ---------------------------------------------------------------------------

CAPTURES_CSV_FIELDNAMES = [
    "campaign_id", "capture_id", "timestamp_iso", "level_index", "variac_vac",
    "pass_id", "quantity", "capture_kind", "node_label", "col_index", "row_index",
    "x_mm", "y_mm", "sequence_index", "n_samples", "mean", "stdev", "three_sigma_pct",
    "min", "max", "duration_s", "sensor_port", "superseded", "note",
]

_CAPTURE_INT_FIELDS = {"level_index", "col_index", "row_index", "sequence_index", "n_samples", "superseded"}
_CAPTURE_FLOAT_FIELDS = {"variac_vac", "x_mm", "y_mm", "mean", "stdev", "three_sigma_pct", "min", "max", "duration_s"}


def new_capture_id() -> str:
    return uuid.uuid4().hex[:8]


def pass_id_for(level_index: int, quantity: str) -> str:
    return f"L{level_index}-{quantity}"


def make_capture_record(
    *,
    campaign_id: str,
    level_index: int,
    variac_vac: float,
    quantity: str,
    capture_kind: str,
    node_label_: str,
    grid: GridConfig,
    sequence_index: int,
    samples: list[float],
    duration_s: float,
    sensor_port: str,
    timestamp_iso: str | None = None,
    note: str = "",
) -> dict:
    """Build one captures.csv row (as a dict) from raw samples."""
    stats = compute_stats(samples)
    col_index, row_index = parse_node_label(node_label_)
    x_mm, y_mm = node_coords_mm(col_index, row_index, grid.pitch_mm)
    return {
        "campaign_id": campaign_id,
        "capture_id": new_capture_id(),
        "timestamp_iso": timestamp_iso or _local_iso_now(),
        "level_index": level_index,
        "variac_vac": variac_vac,
        "pass_id": pass_id_for(level_index, quantity),
        "quantity": quantity,
        "capture_kind": capture_kind,
        "node_label": node_label_,
        "col_index": col_index,
        "row_index": row_index,
        "x_mm": x_mm,
        "y_mm": y_mm,
        "sequence_index": sequence_index,
        "n_samples": stats["n"],
        "mean": stats["mean"],
        "stdev": stats["stdev"],
        "three_sigma_pct": stats["three_sigma_pct"],
        "min": stats["min"],
        "max": stats["max"],
        "duration_s": duration_s,
        "sensor_port": sensor_port,
        "superseded": 0,
        "note": note,
    }


def _local_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z") or time.strftime("%Y-%m-%dT%H:%M:%S")


def append_capture(csv_path, record: dict) -> None:
    """Append one row, writing the header on first creation. Flushed and
    fsynced immediately -- a campaign runs for hours and must survive a
    crash mid-pass (plan section 5)."""
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_file.exists()
    with csv_file.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CAPTURES_CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: _csv_cell(record.get(k, "")) for k in CAPTURES_CSV_FIELDNAMES})
        fh.flush()
        os.fsync(fh.fileno())


def _csv_cell(value):
    return "" if value is None else value


def _read_captures_raw(csv_path) -> list[dict]:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        return []
    with csv_file.open("r", newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _coerce_capture_row(row: dict) -> dict:
    out = dict(row)
    for k in _CAPTURE_INT_FIELDS:
        v = out.get(k, "")
        out[k] = int(float(v)) if v not in (None, "") else None
    for k in _CAPTURE_FLOAT_FIELDS:
        v = out.get(k, "")
        out[k] = float(v) if v not in (None, "") else None
    return out


def read_captures(csv_path) -> list[dict]:
    """All rows, type-coerced. Includes superseded rows -- use
    active_captures() to filter them out for analysis."""
    return [_coerce_capture_row(r) for r in _read_captures_raw(csv_path)]


def active_captures(captures: list[dict]) -> list[dict]:
    return [c for c in captures if not c.get("superseded")]


def mark_superseded(csv_path, capture_id: str, superseded: bool = True) -> bool:
    """Flip the superseded flag on the row with this capture_id (True, the
    default, to retire it on a redo; False to restore it, e.g. undoing an
    accidental redo). CSV is append-only in spirit (no row is ever deleted or
    has its measured values changed); this flag is the one exception, and it
    requires rewriting the file since CSV has no in-place row update. Returns
    True if a row was updated."""
    rows = _read_captures_raw(csv_path)
    changed = False
    for row in rows:
        if row.get("capture_id") == capture_id:
            row["superseded"] = "1" if superseded else "0"
            changed = True
    if not changed:
        return False

    csv_file = Path(csv_path)
    with csv_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CAPTURES_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CAPTURES_CSV_FIELDNAMES})
        fh.flush()
        os.fsync(fh.fileno())
    return True


# ---------------------------------------------------------------------------
# Resume / cursor (plan section 5, campaign.json "cursor")
#
# The capture log is the source of truth: it can never be ahead of what
# actually happened (an append is fsynced before the caller moves on), while
# a cached campaign.json cursor could in principle be stale after a crash
# between the CSV write and the JSON rewrite. Resume always recomputes the
# cursor from captures.csv rather than trusting the JSON blindly.
# ---------------------------------------------------------------------------

QUANTITIES = ("lux", "irr")


def next_cursor(
    captures: list[dict], grid: GridConfig, reference_node: str, current_level_index: int | None = None
) -> dict | None:
    """Given every active (non-superseded) capture in the campaign so far,
    return the next step to take: {"level_index", "pass", "next_kind"
    ("ref_start"/"grid"/"ref_end"), "sequence_index" (-1 for a reference
    capture), "node_label"}. Returns None once the most recent level has
    both passes fully complete (caller then prompts add-level/finish).

    current_level_index lets the caller name the level that should be
    treated as "current" even before its first capture exists (e.g. right
    after the operator enters a new level's VAC, before anything has been
    captured for it yet) -- derived purely from captures, a brand-new level
    would be indistinguishable from "no active level at all"."""
    active = active_captures(captures)
    if current_level_index is None:
        if not active:
            return None
        last_level = max(c["level_index"] for c in active)
    else:
        last_level = current_level_index
        if active:
            last_level = max(last_level, max(c["level_index"] for c in active))

    level_rows = [c for c in active if c["level_index"] == last_level]
    order = serpentine_labels(grid)

    for quantity in QUANTITIES:
        q_rows = [c for c in level_rows if c["quantity"] == quantity]

        if not any(c["capture_kind"] == "ref_start" for c in q_rows):
            return {
                "level_index": last_level, "pass": quantity, "next_kind": "ref_start",
                "sequence_index": -1, "node_label": reference_node,
            }

        done_seq = {c["sequence_index"] for c in q_rows if c["capture_kind"] == "grid"}
        for idx, label in enumerate(order):
            if idx not in done_seq:
                return {
                    "level_index": last_level, "pass": quantity, "next_kind": "grid",
                    "sequence_index": idx, "node_label": label,
                }

        if not any(c["capture_kind"] == "ref_end" for c in q_rows):
            return {
                "level_index": last_level, "pass": quantity, "next_kind": "ref_end",
                "sequence_index": -1, "node_label": reference_node,
            }

    return None


# ---------------------------------------------------------------------------
# Drift normalisation (plan section 6.1)
# ---------------------------------------------------------------------------


def interpolate_reference(ref_start: float, ref_end: float, t_start: float, t_end: float, t: float) -> float:
    if t_end == t_start:
        return ref_start
    fraction = (t - t_start) / (t_end - t_start)
    return ref_start + (ref_end - ref_start) * fraction


def normalize_value(value: float, ref_start: float, ref_interp_t: float) -> float:
    if ref_interp_t == 0:
        return value
    return value * ref_start / ref_interp_t


def drift_pct(ref_start: float | None, ref_end: float | None) -> float | None:
    if not ref_start or ref_end is None:
        return None
    return (ref_end - ref_start) / ref_start * 100.0


def normalize_pass(
    grid_captures: list[dict], ref_start: float, ref_end: float, t_start: float, t_end: float
) -> list[dict]:
    """Return grid_captures with an added 'mean_norm' key, each computed
    from that row's own timestamp against the interpolated reference drift."""
    out = []
    for row in grid_captures:
        t_i = _capture_epoch(row)
        ref_interp_t = interpolate_reference(ref_start, ref_end, t_start, t_end, t_i)
        norm = normalize_value(row["mean"], ref_start, ref_interp_t)
        out.append({**row, "mean_norm": norm})
    return out


def _capture_epoch(row: dict) -> float:
    """timestamp_iso -> epoch seconds. Accepts either a real ISO-8601
    string (as produced by _local_iso_now()) or a bare float (used freely
    by callers/tests that only care about relative ordering)."""
    ts = row.get("timestamp_iso") if isinstance(row, dict) else row
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return time.mktime(time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Correction factor K (plan section 6.3) and level independence (6.4)
# ---------------------------------------------------------------------------


def panel_mean(values_by_node: dict[str, float], covered: list[str]) -> float:
    covered_values = [values_by_node[n] for n in covered if n in values_by_node]
    if not covered_values:
        raise ValueError("no covered-node values available to average")
    return sum(covered_values) / len(covered_values)


def compute_K(panel_mean_value: float, sensor_value: float) -> float:
    if not sensor_value:
        raise ValueError("sensor_value must be nonzero")
    return panel_mean_value / sensor_value


def uniformity_stats(values_by_node: dict[str, float], covered: list[str]) -> dict:
    covered_values = [values_by_node[n] for n in covered if n in values_by_node]
    if not covered_values:
        return {"panel_min": None, "panel_max": None, "uniformity_pct": None, "cv_pct": None}
    lo, hi = min(covered_values), max(covered_values)
    mean = sum(covered_values) / len(covered_values)
    uniformity_pct = (hi - lo) / (hi + lo) * 100.0 if (hi + lo) else None
    if len(covered_values) >= 2 and mean:
        variance = sum((x - mean) ** 2 for x in covered_values) / len(covered_values)
        cv_pct = (variance ** 0.5) / mean * 100.0
    else:
        cv_pct = None
    return {"panel_min": lo, "panel_max": hi, "uniformity_pct": uniformity_pct, "cv_pct": cv_pct}


def k_spread_pct(k_values: list[float]) -> float | None:
    if not k_values:
        return None
    mean_k = sum(k_values) / len(k_values)
    if not mean_k:
        return None
    return (max(k_values) - min(k_values)) / mean_k * 100.0


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Least-squares slope, intercept of ys against xs. None if underdetermined."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    return slope, intercept


def level_independence_report(k_by_level: dict[int, float], variac_by_level: dict[int, float] | None = None) -> dict:
    """Plan section 6.4: recommended K, its spread, a plain-language verdict,
    and (if >= 4 levels) a linear fit of K against VAC."""
    levels = sorted(k_by_level)
    k_values = [k_by_level[lv] for lv in levels]
    spread = k_spread_pct(k_values)
    recommended_k = sum(k_values) / len(k_values) if k_values else None
    single_factor_ok = spread is not None and spread < K_SPREAD_SINGLE_FACTOR_THRESHOLD_PCT

    verdict = (
        f"K is constant to within {spread:.2f}% across {len(levels)} level(s) -- "
        f"one recommended factor ({recommended_k:.4f}) is sufficient."
        if single_factor_ok
        else (
            f"K varies by {spread:.2f}% across {len(levels)} level(s) -- use the per-level "
            "table rather than a single factor."
            if spread is not None
            else "Not enough levels to assess K spread."
        )
    )

    fit = None
    if variac_by_level and len(levels) >= MIN_LEVELS_FOR_LINEAR_FIT:
        xs = [variac_by_level[lv] for lv in levels if lv in variac_by_level]
        ys = [k_by_level[lv] for lv in levels if lv in variac_by_level]
        fit = linear_fit(xs, ys)

    return {
        "levels": levels,
        "k_by_level": dict(k_by_level),
        "recommended_k": recommended_k,
        "k_spread_pct": spread,
        "single_factor_sufficient": single_factor_ok,
        "verdict": verdict,
        "linear_fit_vac": fit,  # (slope, intercept) or None
    }


# ---------------------------------------------------------------------------
# Lux/irradiance ratio map (plan section 6.5)
# ---------------------------------------------------------------------------


def lux_irr_ratio_map(lux_by_node: dict[str, float], irr_by_node: dict[str, float]) -> dict[str, float]:
    return {
        node: lux_by_node[node] / irr_by_node[node]
        for node in lux_by_node
        if node in irr_by_node and irr_by_node[node]
    }


# ---------------------------------------------------------------------------
# correction_factors.csv (plan section 6.6)
# ---------------------------------------------------------------------------

CORRECTION_FACTORS_CSV_FIELDNAMES = [
    "campaign_id", "level_index", "variac_vac", "quantity", "pass_id", "drift_pct",
    "drift_flagged", "panel_mean", "sensor_value", "K", "panel_min", "panel_max",
    "uniformity_pct", "cv_pct", "n_covered_nodes", "plane_min", "plane_max",
]


def write_correction_factors_csv(csv_path, rows: list[dict]) -> None:
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with csv_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CORRECTION_FACTORS_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(row.get(k, "")) for k in CORRECTION_FACTORS_CSV_FIELDNAMES})


def read_correction_factors_csv(csv_path) -> list[dict]:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        return []
    with csv_file.open("r", newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


# ---------------------------------------------------------------------------
# campaign.json (plan section 5)
# ---------------------------------------------------------------------------


def new_campaign_id(slug: str, when: float | None = None) -> str:
    ts = time.localtime(when)
    stamp = time.strftime("%Y%m%d-%H%M%S", ts)
    safe_slug = "".join(c if (c.isalnum() or c in "-_") else "-" for c in slug).strip("-") or "campaign"
    return f"cal-{stamp}-{safe_slug}"


def default_campaign_config(
    *,
    campaign_id: str,
    operator: str,
    grid: GridConfig = GridConfig(),
    reference_node: str = DEFAULT_REFERENCE_NODE,
    dwell_seconds: float = DEFAULT_DWELL_SECONDS,
    warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
    drift_flag_pct: float = DEFAULT_DRIFT_FLAG_PCT,
    panel: dict | None = None,
    sensor_node: str = DEFAULT_REFERENCE_NODE,
    sensor_head_height_mm: float = 12.0,
    lamp_note: str = "",
    created_at: str | None = None,
) -> dict:
    return {
        "campaign_id": campaign_id,
        "created_at": created_at or _local_iso_now(),
        "operator": operator,
        "grid": {
            "cols": grid.cols,
            "rows": grid.rows,
            "pitch_mm": grid.pitch_mm,
            "col_labels": [col_letter(i) for i in range(grid.cols)],
            "origin_note": "A1 = front-left intersection facing the box",
        },
        "reference_node": reference_node,
        "dwell_seconds": dwell_seconds,
        "warmup_seconds": warmup_seconds,
        "drift_flag_pct": drift_flag_pct,
        "panel": panel or {},
        "sensor_node": sensor_node,
        "sensor_head_height_mm": sensor_head_height_mm,
        "lamp_note": lamp_note,
        "levels": [],
        "cursor": None,
    }


def grid_from_campaign(config: dict) -> GridConfig:
    g = config.get("grid", {})
    return GridConfig(
        cols=g.get("cols", DEFAULT_COLS),
        rows=g.get("rows", DEFAULT_ROWS),
        pitch_mm=g.get("pitch_mm", DEFAULT_PITCH_MM),
    )


def save_campaign_config(campaign_dir, config: dict) -> None:
    campaign_dir = Path(campaign_dir)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    path = campaign_dir / "campaign.json"
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    tmp_path.replace(path)


def load_campaign_config(campaign_dir) -> dict:
    path = Path(campaign_dir) / "campaign.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def captures_csv_path(campaign_dir) -> Path:
    return Path(campaign_dir) / "captures.csv"


def analysis_dir(campaign_dir) -> Path:
    return Path(campaign_dir) / "analysis"


# ---------------------------------------------------------------------------
# Full-campaign analysis orchestration (plan section 6) -- everything the
# Analysis tab needs, computed here so the GUI layer stays a thin renderer.
# ---------------------------------------------------------------------------


def effective_covered_nodes(config: dict, grid: GridConfig) -> list[str]:
    panel = config.get("panel") or {}
    manual = panel.get("covered_nodes")
    if manual:
        return list(manual)
    width = panel.get("width_mm")
    height = panel.get("height_mm")
    if not width or not height:
        return []
    return covered_nodes(grid, width, height)


def analyze_campaign(captures: list[dict], config: dict) -> dict:
    """Everything derivable from a campaign's captures: per-level/quantity
    raw+normalised node maps, drift, correction factors K, uniformity, the
    level-independence verdict, and the lux/irr ratio map. Incomplete passes
    (missing a reference capture or any grid node) are skipped rather than
    guessed at."""
    grid = grid_from_campaign(config)
    reference_node = config.get("reference_node", DEFAULT_REFERENCE_NODE)
    sensor_node = config.get("sensor_node", reference_node)
    drift_flag_pct = config.get("drift_flag_pct", DEFAULT_DRIFT_FLAG_PCT)
    covered = effective_covered_nodes(config, grid)

    active = active_captures(captures)
    levels = sorted({c["level_index"] for c in active})

    levels_out: dict[int, dict] = {}
    correction_rows: list[dict] = []
    k_by_level: dict[str, dict[int, float]] = {q: {} for q in QUANTITIES}
    variac_by_level: dict[int, float] = {}

    for level_index in levels:
        level_active = [c for c in active if c["level_index"] == level_index]
        if level_active:
            variac_by_level[level_index] = level_active[0]["variac_vac"]
        levels_out[level_index] = {}

        for quantity in QUANTITIES:
            q_rows = [c for c in level_active if c["quantity"] == quantity]
            ref_start_row = next((c for c in q_rows if c["capture_kind"] == "ref_start"), None)
            ref_end_row = next((c for c in q_rows if c["capture_kind"] == "ref_end"), None)
            grid_rows = [c for c in q_rows if c["capture_kind"] == "grid" and c.get("mean") is not None]
            if (
                ref_start_row is None or ref_end_row is None or not grid_rows
                or ref_start_row.get("mean") is None or ref_end_row.get("mean") is None
            ):
                continue

            ref_start_val, ref_end_val = ref_start_row["mean"], ref_end_row["mean"]
            t_start, t_end = _capture_epoch(ref_start_row), _capture_epoch(ref_end_row)
            d_pct = drift_pct(ref_start_val, ref_end_val)
            drift_flagged = d_pct is not None and abs(d_pct) > drift_flag_pct

            normed = normalize_pass(grid_rows, ref_start_val, ref_end_val, t_start, t_end)
            raw_by_node = {r["node_label"]: r["mean"] for r in grid_rows}
            norm_by_node = {r["node_label"]: r["mean_norm"] for r in normed}

            pm = panel_mean(norm_by_node, covered) if covered else None
            sensor_value = norm_by_node.get(sensor_node)
            k = compute_K(pm, sensor_value) if (pm is not None and sensor_value) else None
            uni = uniformity_stats(norm_by_node, covered) if covered else {}
            n_covered = len([n for n in covered if n in norm_by_node])
            plane_values = list(norm_by_node.values())

            levels_out[level_index][quantity] = {
                "raw_by_node": raw_by_node,
                "norm_by_node": norm_by_node,
                "ref_start": ref_start_val,
                "ref_end": ref_end_val,
                "drift_pct": d_pct,
                "drift_flagged": drift_flagged,
                "panel_mean": pm,
                "sensor_value": sensor_value,
                "K": k,
                "n_covered_nodes": n_covered,
                "plane_min": min(plane_values) if plane_values else None,
                "plane_max": max(plane_values) if plane_values else None,
                **uni,
            }

            if k is not None:
                k_by_level[quantity][level_index] = k

            correction_rows.append({
                "campaign_id": config.get("campaign_id", ""),
                "level_index": level_index,
                "variac_vac": variac_by_level.get(level_index),
                "quantity": quantity,
                "pass_id": pass_id_for(level_index, quantity),
                "drift_pct": d_pct,
                "drift_flagged": int(drift_flagged),
                "panel_mean": pm,
                "sensor_value": sensor_value,
                "K": k,
                "panel_min": uni.get("panel_min"),
                "panel_max": uni.get("panel_max"),
                "uniformity_pct": uni.get("uniformity_pct"),
                "cv_pct": uni.get("cv_pct"),
                "n_covered_nodes": n_covered,
                "plane_min": min(plane_values) if plane_values else None,
                "plane_max": max(plane_values) if plane_values else None,
            })

    independence = {
        quantity: level_independence_report(k_by_level[quantity], variac_by_level)
        for quantity in QUANTITIES
        if k_by_level[quantity]
    }

    ratio_maps = {
        level_index: lux_irr_ratio_map(
            levels_out[level_index]["lux"]["norm_by_node"], levels_out[level_index]["irr"]["norm_by_node"]
        )
        for level_index in levels
        if "lux" in levels_out[level_index] and "irr" in levels_out[level_index]
    }

    return {
        "grid": grid,
        "reference_node": reference_node,
        "sensor_node": sensor_node,
        "covered_nodes": covered,
        "levels": levels_out,
        "variac_by_level": variac_by_level,
        "k_by_level": k_by_level,
        "independence": independence,
        "ratio_maps": ratio_maps,
        "correction_rows": correction_rows,
    }
