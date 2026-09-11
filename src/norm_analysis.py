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
for X in {Wp, Vp, Ip, Isc}. A "group" is (cell_type, cells_series,
cells_parallel) from the panel config file, keyed by each panel name's
*serial prefix* (e.g. "P124" out of "P124N042") rather than the full name,
so the operator doesn't have to enumerate every individual unit.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

DEFAULT_TOLERANCE = 0.10  # +/-10% of 1000 W/m^2
REFERENCE_IRRADIANCE = 1000.0

_SERIAL_PREFIX_PATTERN = re.compile(r"^([A-Za-z]+\d*)")

_ROW_FIELDS = ("Wp", "Vp", "Ip", "Isc")


def extract_serial_prefix(panel_name: str) -> str:
    """Extract the leading serial prefix (letters then optional digits,
    e.g. "P124" out of "P124N042", "IXYS" out of "IXYS123") and uppercase
    it for case-insensitive matching (e.g. "p124N042" -> "P124").

    Permanent fallback, not just a historical-data accommodation: any name
    that doesn't fit the pattern (old junk data, a typo, a one-off test
    panel, anything unanticipated) must never crash or throw -- it just
    becomes its own standalone prefix (the whole name, uppercased), which
    naturally won't match any config entry and is silently excluded from
    the analysis rather than being a special error case.
    """
    name = (panel_name or "").strip()
    if not name:
        return ""
    match = _SERIAL_PREFIX_PATTERN.match(name)
    if match:
        return match.group(1).upper()
    return name.upper()


def load_panel_config(path) -> dict:
    """Read a CSV of serial_prefix,cell_type,cells_series,cells_parallel
    into {serial_prefix (uppercased): {"cell_type", "cells_series",
    "cells_parallel"}}. Tolerant of a missing/malformed file -- returns {}
    rather than raising, since the panel config is optional until the
    operator has created one."""
    config = {}
    csv_path = Path(path)
    if not csv_path.exists():
        return config

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                prefix = (row.get("serial_prefix") or "").strip()
                if not prefix or prefix.startswith("#"):
                    continue
                try:
                    cells_series = int(float(row.get("cells_series", "").strip()))
                    cells_parallel = int(float(row.get("cells_parallel", "").strip()))
                except (TypeError, ValueError):
                    continue
                config[prefix.upper()] = {
                    "cell_type": (row.get("cell_type") or "").strip(),
                    "cells_series": cells_series,
                    "cells_parallel": cells_parallel,
                }
    except Exception:
        return {}

    return config


def save_panel_config(path, config: dict) -> None:
    """Write {serial_prefix: {"cell_type", "cells_series", "cells_parallel"}}
    back out as a plain CSV (sorted by prefix for a stable diff), overwriting
    whatever was there -- used by the in-app Panel Config editor tab so every
    table edit stays persisted to disk."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["serial_prefix", "cell_type", "cells_series", "cells_parallel"])
        for prefix in sorted(config.keys()):
            entry = config[prefix]
            writer.writerow([prefix, entry["cell_type"], entry["cells_series"], entry["cells_parallel"]])


def group_key_for(panel_name: str, config: dict):
    """Resolve a panel name to its (cell_type, cells_series, cells_parallel)
    group key via its serial prefix, or None if that prefix has no config
    entry (the measurement is excluded from analysis, not crashed on)."""
    prefix = extract_serial_prefix(panel_name)
    entry = config.get(prefix)
    if entry is None:
        return None
    return (entry["cell_type"], entry["cells_series"], entry["cells_parallel"])


def group_label(group_key) -> str:
    cell_type, series, parallel = group_key
    return f"{cell_type} {series}S{parallel}P" if cell_type else f"{series}S{parallel}P"


def _parse_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_norm_dataset(rows, config: dict, tolerance: float = DEFAULT_TOLERANCE) -> dict:
    """Compute normalized Wp/Vp/Ip/Isc for every measurement row that
    belongs to a group with an established 1000 W/m^2 reference.

    Always recomputed fresh from the full current dataset (rows, config) --
    no incremental/cached state to go stale, which is cheap at the data
    volumes this app produces.

    Returns:
        {
            "groups": {group_key: {"label": str, "points": [
                {"irradiance": float, "panel_name": str,
                 "norm_wp": float, "norm_vp": float,
                 "norm_ip": float, "norm_isc": float}, ...
            ]}},
            "skipped_no_config": int,   # panel name's prefix has no config entry
            "skipped_bad_data": int,    # missing/unparseable Wp/Vp/Ip/Isc/irradiance
            "skipped_no_reference": int,  # valid group, but no reference established yet
        }
    """
    lower = REFERENCE_IRRADIANCE * (1.0 - tolerance)
    upper = REFERENCE_IRRADIANCE * (1.0 + tolerance)

    parsed_by_group: dict = {}
    skipped_no_config = 0
    skipped_bad_data = 0

    for row in rows:
        panel_name = row.get("panel_name", "")
        group_key = group_key_for(panel_name, config)
        if group_key is None:
            skipped_no_config += 1
            continue

        irradiance = _parse_float(row.get("irradiance_measured_live"))
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

        groups[group_key] = {"label": group_label(group_key), "points": points}

    return {
        "groups": groups,
        "skipped_no_config": skipped_no_config,
        "skipped_bad_data": skipped_bad_data,
        "skipped_no_reference": skipped_no_reference,
    }
