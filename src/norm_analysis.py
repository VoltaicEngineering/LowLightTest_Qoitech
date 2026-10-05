#!/usr/bin/env python3
"""Normalized low-light performance analysis — pure logic, no Qt/UI.

Kept separate from low_light_app.py so the math is testable in isolation,
matching the existing project split (IV_curve_CURRENT_V3.py = hardware/
domain logic, low_light_app.py = GUI orchestration).

Methodology (confirmed with the operator, since the source Google Slides
deck wasn't directly viewable):
    Ref_X[group] = average(X over all group measurements with
                            irradiance_measured_live within +/-tolerance
                            of 1000 W/m^2 -- 900 <= ... <= 1100 at the
                            default 10% tolerance, adjustable in Settings)
    Norm_X (%)   = X_measured / Ref_X[group] * 100
for X in {Wp, Vp, Ip, Isc}. In lightbox mode the setpoint
(lightbox_setpoint_wm2) stands in for irradiance_measured_live, both to pick
the reference rows and as the x value -- the sensor sits at one grid node, so
its reading is not the panel average the setpoint was set for. A "group" is one panel, identified by the full
Panel Name entered on the Measure tab (matched case-insensitively, ignoring
surrounding whitespace). Every panel is normalized against its own 1000 W/m^2
measurements, and its cell_type / cells_series / cells_parallel come from its
own row in the panel config file -- two units sharing a serial prefix (e.g.
"P124N042" and "P124N043") no longer pool into one baseline.
"""
from __future__ import annotations

import csv
from pathlib import Path

DEFAULT_TOLERANCE = 0.10  # +/-10% of 1000 W/m^2
REFERENCE_IRRADIANCE = 1000.0

_ROW_FIELDS = ("Wp", "Vp", "Ip", "Isc")
MEASURED_IRRADIANCE_FIELD = "irradiance_measured_live"
SETPOINT_IRRADIANCE_FIELD = "lightbox_setpoint_wm2"


def normalize_panel_name(panel_name: str) -> str:
    """Matching key for a panel name: surrounding whitespace stripped and
    case-folded, so "p124n042 " and "P124N042" are the same panel. Any
    string is accepted -- a name with no config entry is simply excluded
    from the analysis, never an error."""
    return (panel_name or "").strip().casefold()


def parse_dimension_mm(text):
    """Optional panel dimension: blank -> None, else a positive float.
    Raises ValueError for anything else."""
    text = (text or "").strip()
    if not text:
        return None
    value = float(text)
    if not value > 0:
        raise ValueError(f"dimension must be > 0 mm, got {text!r}")
    return value


def load_panel_config(path) -> dict:
    """Read a CSV of panel_name,cell_type,cells_series,cells_parallel,
    width_mm,height_mm into {panel_name (as written): {"cell_type",
    "cells_series", "cells_parallel", "width_mm", "height_mm"}}. Width and
    height are optional (None when blank or missing, e.g. an older file) and
    only needed for lightbox runs. Tolerant of a missing/malformed file -- returns {}
    rather than raising, since the panel config is optional until the
    operator has created one. A legacy file with a serial_prefix column is
    still read (each prefix becomes a panel name), so nothing is lost; those
    rows just won't match until they're renamed to full panel names."""
    config = {}
    csv_path = Path(path)
    if not csv_path.exists():
        return config

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name = (row.get("panel_name") or row.get("serial_prefix") or "").strip()
                if not name or name.startswith("#"):
                    continue
                try:
                    cells_series = int(float(row.get("cells_series", "").strip()))
                    cells_parallel = int(float(row.get("cells_parallel", "").strip()))
                except (AttributeError, TypeError, ValueError):
                    continue
                try:
                    width_mm = parse_dimension_mm(row.get("width_mm"))
                    height_mm = parse_dimension_mm(row.get("height_mm"))
                except ValueError:
                    width_mm = height_mm = None
                config[name] = {
                    "cell_type": (row.get("cell_type") or "").strip(),
                    "cells_series": cells_series,
                    "cells_parallel": cells_parallel,
                    "width_mm": width_mm,
                    "height_mm": height_mm,
                }
    except Exception:
        return {}

    return config


def save_panel_config(path, config: dict) -> None:
    """Write {panel_name: {"cell_type", "cells_series", "cells_parallel",
    "width_mm", "height_mm"}}
    back out as a plain CSV (sorted by name for a stable diff), overwriting
    whatever was there -- used by the in-app Panel Config editor tab so every
    table edit stays persisted to disk."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["panel_name", "cell_type", "cells_series", "cells_parallel", "width_mm", "height_mm"])
        for name in sorted(config.keys(), key=normalize_panel_name):
            entry = config[name]
            dims = ["" if entry.get(k) is None else f"{entry[k]:g}" for k in ("width_mm", "height_mm")]
            writer.writerow([name, entry["cell_type"], entry["cells_series"], entry["cells_parallel"], *dims])


def lookup_panel(panel_name: str, config: dict):
    """(configured name, entry) for a panel name, matched case-insensitively,
    or None."""
    return config_index(config).get(normalize_panel_name(panel_name))


def config_index(config: dict) -> dict:
    """{normalized panel name: (configured name, entry)} for lookups."""
    return {normalize_panel_name(name): (name, entry) for name, entry in config.items()}


def group_label(name: str, entry: dict) -> str:
    cell_type, series, parallel = entry["cell_type"], entry["cells_series"], entry["cells_parallel"]
    layout = f"{cell_type} {series}S{parallel}P" if cell_type else f"{series}S{parallel}P"
    return f"{name} ({layout})"


def _parse_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_norm_dataset(
    rows,
    config: dict,
    tolerance: float = DEFAULT_TOLERANCE,
    irradiance_field: str = MEASURED_IRRADIANCE_FIELD,
) -> dict:
    """Compute normalized Wp/Vp/Ip/Isc for every measurement row that
    belongs to a group with an established 1000 W/m^2 reference.

    irradiance_field is the summary-CSV column used as each row's irradiance
    (reference selection and x value): MEASURED_IRRADIANCE_FIELD normally,
    SETPOINT_IRRADIANCE_FIELD in lightbox mode. Rows with it blank count as
    skipped_bad_data.

    Always recomputed fresh from the full current dataset (rows, config) --
    no incremental/cached state to go stale, which is cheap at the data
    volumes this app produces.

    Returns:
        {
            "groups": {normalized panel name: {
                "label": str, "panel_name": str,  # configured spelling
                "cell_type": str, "cells_series": int, "cells_parallel": int,
                "points": [
                {"irradiance": float, "panel_name": str,
                 "norm_wp": float, "norm_vp": float,
                 "norm_ip": float, "norm_isc": float}, ...
            ]}},
            "skipped_no_config": int,   # panel name has no config entry
            "skipped_bad_data": int,    # missing/unparseable Wp/Vp/Ip/Isc/irradiance
            "skipped_no_reference": int,  # valid group, but no reference established yet
        }
    """
    lower = REFERENCE_IRRADIANCE * (1.0 - tolerance)
    upper = REFERENCE_IRRADIANCE * (1.0 + tolerance)

    parsed_by_group: dict = {}
    skipped_no_config = 0
    skipped_bad_data = 0

    index = config_index(config)

    for row in rows:
        panel_name = row.get("panel_name", "")
        group_key = normalize_panel_name(panel_name)
        if group_key not in index:
            skipped_no_config += 1
            continue

        irradiance = _parse_float(row.get(irradiance_field))
        values = {field: _parse_float(row.get(field)) for field in _ROW_FIELDS}
        if irradiance is None or irradiance <= 0 or any(v is None for v in values.values()):
            skipped_bad_data += 1
            continue

        parsed_by_group.setdefault(group_key, []).append(
            {"panel_name": panel_name, "irradiance": irradiance, **values}
        )

    groups = {}
    skipped_no_reference = 0

    for group_key, entries in parsed_by_group.items():
        qualifying = [e for e in entries if lower <= e["irradiance"] <= upper]
        if not qualifying:
            skipped_no_reference += len(entries)
            continue

        ref = {
            field: sum(e[field] for e in qualifying) / len(qualifying)
            for field in _ROW_FIELDS
        }
        if any(ref[field] == 0 for field in _ROW_FIELDS):
            skipped_no_reference += len(entries)
            continue

        points = []
        for e in entries:
            points.append({
                "irradiance": e["irradiance"],
                "panel_name": e["panel_name"],
                "norm_wp": e["Wp"] / ref["Wp"] * 100.0,
                "norm_vp": e["Vp"] / ref["Vp"] * 100.0,
                "norm_ip": e["Ip"] / ref["Ip"] * 100.0,
                "norm_isc": e["Isc"] / ref["Isc"] * 100.0,
            })
        points.sort(key=lambda p: p["irradiance"])

        name, entry = index[group_key]
        groups[group_key] = {
            "label": group_label(name, entry),
            "panel_name": name,
            "cell_type": entry["cell_type"],
            "cells_series": entry["cells_series"],
            "cells_parallel": entry["cells_parallel"],
            "points": points,
        }

    return {
        "groups": groups,
        "skipped_no_config": skipped_no_config,
        "skipped_bad_data": skipped_bad_data,
        "skipped_no_reference": skipped_no_reference,
    }
