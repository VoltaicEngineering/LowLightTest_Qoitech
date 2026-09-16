# Build Plan: Lightbox Spatial Calibration Tool

**Repo:** `33_LowLightTestingGodHelpme`
**Author:** AK (spec drafted with Claude, 2026-09-16)
**Status:** Ready to build
**Audience:** Claude Code, working in this repo

---

## 1. Purpose

The indoor lightbox is a steel cart with an array of halogen MR16 lamps under a
diffuser, topped by a clear acrylic sheet that is the panel test plane. Lamp
brightness is set with a Circuit Specialists TDGC-2KM variac (0 to 130 VAC,
2000 VA) with a digital AC voltage readout.

Illumination across that plane is not uniform. Today a panel under test occupies
the middle of the plane while the lux meter and irradiance sensor sit wherever
there is room, typically off toward a corner. The number we log as "the
irradiance during this sweep" is therefore a reading from a place the panel is
not.

This tool collects the spatial map that closes that gap: at each node of the
scored 5 cm grid, and at each variac setting, record lux and irradiance with
their dispersion. The map yields a correction factor **K** per variac level:

```
K = mean(quantity over the panel footprint nodes) / quantity(at the sensor node)
```

During a normal IV run the operator then reads the sensor at its fixed offset
position and multiplies by K to get the panel-average value.

A secondary output falls out for free: the map tells you how uniform the box
actually is over the panel footprint, which is a number the test reports have
been missing.

---

## 2. Physical setup and conventions

| Item | Value |
|---|---|
| Measurement plane | Top acrylic surface, where the panel sits |
| Grid pitch | 50 mm |
| Grid size | 11 columns x 7 rows = **77 nodes** |
| Column labels | A through K (left to right, facing the box) |
| Row labels | 1 through 7 (front to back) |
| Node span | 500 mm x 300 mm |
| Origin | Node **A1** = front-left intersection |
| Lux meter | Triplett LT68, USB bridge CP210x |
| Irradiance sensor | serial, USB bridge CH9102, prints `Irradiance: <float>` |
| Brightness control | variac, operator enters the displayed VAC |

Node A1 is the anchor. Every node also carries derived mm coordinates
(`x_mm = (col_index) * 50`, `y_mm = (row_index) * 50`) so the analysis code can
interpolate and plot without parsing labels.

---

## 3. Campaign structure

A campaign is a nested loop. Variac level is the outer loop because changing the
variac costs a settle wait and perturbs lamp temperature, while moving a sensor
costs seconds.

```
Campaign
└── Level (operator enters VAC; loop repeats until operator says "done")
    ├── Warm-up countdown after the variac is changed
    ├── Pass: LUX
    │   ├── Reference node capture  (start)
    │   ├── 77 grid node captures   (serpentine order)
    │   └── Reference node capture  (end)
    └── Pass: IRR
        ├── Reference node capture  (start)
        ├── 77 grid node captures
        └── Reference node capture  (end)
```

**Levels are open-ended.** The operator does not declare the level list up front.
After finishing both passes at a level, the app asks: add another level, or
finish the campaign. If another, the operator sets the variac, types the new
VAC, and the warm-up countdown starts. This keeps a session interruptible and
lets the operator stop when the data looks sufficient.

**Passes are separate** because there is one sensor jig and the two heads are not
co-located. Run LUX and IRR back to back at the same level so lamp thermal state
is as close as it can be between them.

**Capture** is manually triggered per node. On trigger the app collects samples
for **3.0 s** (configurable), then reports n, mean, stdev and 3σ/mean %, and
auto-advances to the next node.

**Serpentine order** (left to right on row 1, right to left on row 2, and so on)
so the operator never walks the jig back across the plate. Order is computed,
displayed, and stored per capture as `sequence_index`.

**Reference node** defaults to the centre node **F4** and is configurable. It is
captured at the start and end of every pass. The app computes
`drift_pct = (ref_end - ref_start) / ref_start * 100` and flags the pass if
`|drift_pct|` exceeds a threshold (default 3 %).

---

## 4. Where the code goes

AK's call: a **standalone GUI** in this repo that shares the existing hardware
modules. No new tab in `low_light_app.py`.

```
src/lightbox_calibration.py     <- new standalone PyQt6 app, entry point
src/calibration_model.py        <- pure logic: grid, ordering, stats, K maths, CSV IO
src/sensors.py                  <- (Phase 0, see below) extracted shared sensor layer
tests/test_calibration_model.py <- unit tests, no hardware
docs/lightbox-calibration-plan.md  <- this file
```

Keep all measurement and analysis maths in `calibration_model.py` with zero Qt
imports, so it is unit-testable headlessly.

### Phase 0 (recommended, ~150 lines, behaviour-preserving)

The pieces the calibration app needs are currently inside `low_light_app.py`,
which means importing them drags in the whole Qt measurement application:

- `detect_sensor_ports()`, `_LUX_DEVICE_PATTERN`, `_IRR_DEVICE_PATTERN`
- `LiveValue`, `TimeSeriesBuffer`, `HistoryIrradiance`
- `lux_poll_loop()`
- `load_cached_port()` / `save_cached_port()`
- constants `LUX_TIMEOUT_S`, `LUX_SETTLE_S`, `LUX_POLL_INTERVAL_S`,
  `IRR_BAUD_DEFAULT`, `IRR_WINDOW_S`, `LIVE_STALE_S`

Move them verbatim into `src/sensors.py` and have `low_light_app.py` re-import
them (`from sensors import detect_sensor_ports, LiveValue, ...`). Nothing in the
measurement app's behaviour changes, and the calibration app gets a clean
dependency. `listen.py` and `triplett_lt68_probe_v3.py` stay untouched and are
imported by `sensors.py` exactly as they are today.

If AK prefers to skip Phase 0, the fallback is `from low_light_app import ...`,
which works but pulls in Qt widget construction at import time and couples the
two apps. Flag this in the PR rather than deciding silently.

### Port exclusivity

Windows COM ports are exclusive. `low_light_app.py` and
`lightbox_calibration.py` cannot both hold the sensors. The calibration app must
detect a busy port and say so plainly ("COM6 is in use, close the Low Light
Testing app first") instead of retrying in a backoff loop.

---

## 5. Data model

### On-disk layout

```
calibration/
└── <campaign_id>/                 e.g. cal-20260916-142200-lightbox
    ├── campaign.json              config + progress cursor
    ├── captures.csv               append-only, one row per capture
    ├── analysis/
    │   ├── correction_factors.csv
    │   ├── heatmap_L1_lux.png
    │   ├── heatmap_L1_irr.png
    │   └── ...
    └── notes.md                   free-text operator log, optional
```

`campaign_id` format: `cal-YYYYMMDD-HHMMSS-<slug>`.

### campaign.json

```json
{
  "campaign_id": "cal-20260916-142200-lightbox",
  "created_at": "2026-09-16T14:22:00-04:00",
  "operator": "AK",
  "grid": {
    "cols": 11, "rows": 7, "pitch_mm": 50,
    "col_labels": ["A","B","C","D","E","F","G","H","I","J","K"],
    "origin_note": "A1 = front-left intersection facing the box"
  },
  "reference_node": "F4",
  "dwell_seconds": 3.0,
  "warmup_seconds": 120,
  "drift_flag_pct": 3.0,
  "panel": {
    "name": "P124R5B",
    "width_mm": 200, "height_mm": 150,
    "anchor": "centered",
    "covered_nodes": ["E3","F3","G3","E4","F4","G4","E5","F5","G5"]
  },
  "sensor_node": "B6",
  "sensor_head_height_mm": 12,
  "lamp_note": "halogen MR16 array, diffuser in place",
  "levels": [
    {"level_index": 1, "variac_vac": 42.0, "started_at": "...",
     "passes": {"lux": "complete", "irr": "complete"}}
  ],
  "cursor": {"level_index": 2, "pass": "lux", "sequence_index": 31}
}
```

`campaign.json` is rewritten after every capture. `cursor` is what makes the
session resumable after a crash, a COM dropout or a lunch break, which matters
because a full campaign is hours long. The repo already uses this pattern in
`report/iv_campaign_state.json`.

### captures.csv

Append-only long format, one row per capture, header written on create. Flush
and fsync after every row.

| Column | Type | Notes |
|---|---|---|
| `campaign_id` | str | |
| `capture_id` | str | uuid4 hex, 8 chars |
| `timestamp_iso` | str | local time with offset |
| `level_index` | int | 1-based |
| `variac_vac` | float | as entered by the operator |
| `pass_id` | str | `L1-lux`, `L1-irr` |
| `quantity` | str | `lux` or `irr` |
| `capture_kind` | str | `grid`, `ref_start`, `ref_end` |
| `node_label` | str | `F4` |
| `col_index` | int | 0-based |
| `row_index` | int | 0-based |
| `x_mm` | float | |
| `y_mm` | float | |
| `sequence_index` | int | position in the serpentine order, -1 for reference captures |
| `n_samples` | int | |
| `mean` | float | lx or W/m2 |
| `stdev` | float | blank if n < 2 |
| `three_sigma_pct` | float | `3*stdev/mean*100` |
| `min` | float | |
| `max` | float | |
| `duration_s` | float | actual elapsed, not nominal |
| `sensor_port` | str | `COM6` |
| `superseded` | int | `1` if a later re-capture of the same node replaced it |
| `note` | str | free text |

A redo writes a **new row** and sets `superseded=1` on the old one rather than
editing in place. Nothing is ever deleted. The analysis layer filters
`superseded == 0`.

---

## 6. Analysis maths

All of this lives in `calibration_model.py`, pure functions over the parsed
captures.

### 6.1 Drift normalisation

For each pass, with `t_start` and `t_end` the reference capture times and `t_i`
each grid capture time:

```
ref_interp(t) = ref_start + (ref_end - ref_start) * (t - t_start) / (t_end - t_start)
value_norm(i) = value(i) * ref_start / ref_interp(t_i)
```

Store both raw and normalised values in the analysis output. If
`|drift_pct| <= drift_flag_pct` the two are near-identical and it costs nothing;
if drift was real, the normalised map is the usable one. Report drift_pct per
pass in `correction_factors.csv` so it is never invisible.

### 6.2 Panel footprint node set

Given panel width W and height H in mm, centred on the grid:

```
x_centre = (cols - 1) * pitch / 2      # 250 mm for 11 cols at 50 mm
y_centre = (rows - 1) * pitch / 2      # 150 mm for 7 rows at 50 mm
covered = { node : |x_mm - x_centre| <= W/2 and |y_mm - y_centre| <= H/2 }
```

Guard: if `covered` has fewer than 4 nodes, warn that the panel is smaller than
the grid pitch can resolve and that the panel-average will be a poor estimate.
Also support a manual override where the operator clicks nodes on the grid
widget to define the footprint directly.

### 6.3 Correction factor

Per level, per quantity:

```
panel_mean   = mean(value_norm over covered nodes)
sensor_value = value_norm at sensor_node
K            = panel_mean / sensor_value
```

Plus uniformity statistics over the covered nodes:

```
uniformity_pct = (max - min) / (max + min) * 100     # standard non-uniformity
cv_pct         = stdev / mean * 100
```

### 6.4 Level independence check

K is a ratio of two points in the same light field. If the lamp array's spatial
shape is fixed, K should be constant across variac levels and a single number
can be used everywhere. The analysis must test this rather than assume it:

```
K_spread_pct = (max(K over levels) - min(K over levels)) / mean(K) * 100
```

Report a single recommended K with its spread. If `K_spread_pct < 2`, state that
one factor is sufficient. If it is larger, output a per-level K table and, if
there are at least 4 levels, a linear fit of K against VAC.

### 6.5 Lux-to-irradiance ratio map

At each node compute `lux_mean / irr_mean` per level. This is worth plotting
because it exposes spectral non-uniformity across the plane, and comparing the
ratio across levels exposes the halogen colour-temperature shift (see section 9).

### 6.6 correction_factors.csv

One row per (level, quantity):

`campaign_id, level_index, variac_vac, quantity, pass_id, drift_pct, drift_flagged, panel_mean, sensor_value, K, panel_min, panel_max, uniformity_pct, cv_pct, n_covered_nodes, plane_min, plane_max`

---

## 7. GUI specification

PyQt6. Reuse `_STYLESHEET` and `_STAGE_STYLE` from `low_light_app.py` verbatim so
the two apps look like one toolchain. Copy them into `sensors.py` or a small
`ui_style.py`; do not fork the colours.

### 7.1 Setup tab

- Campaign name / slug, operator name, output folder (default `<repo>/calibration`).
- Grid config: cols (11), rows (7), pitch mm (50). Editable, defaults as shown.
- Reference node picker (default F4).
- Dwell seconds (3.0), warm-up seconds (120), drift flag % (3.0).
- Panel: name, width mm, height mm. Live preview highlights the covered nodes on
  a grid widget as the numbers are typed. A "pick nodes manually" toggle switches
  to click-to-select.
- Sensor node: click a node on the same grid widget. This is where the sensor
  jig sits during normal IV runs.
- Sensor head height mm, lamp note, free-text campaign note.
- Buttons: **New campaign**, **Resume campaign** (folder picker, reads
  `campaign.json`, restores the cursor).

### 7.2 Sensors tab

Straight lift of the pattern already in `low_light_app._build_ui` / `_connect_sensors`:

- Port combos for lux and irr, populated from `serial.tools.list_ports`, with
  auto-selection from `detect_sensor_ports()` falling back to the cached port.
- Connect / Refresh Ports buttons, auto-connect on launch.
- Live readouts with the live / error / stale badges and the 600 s timeseries
  plot. The timeseries is genuinely useful here: it shows the operator when the
  lamps have settled after a variac change.
- Clear message when a port is held by another process.

### 7.3 Run tab (the main screen)

Header strip, always visible:

```
Level 3 of 3   |   42.0 VAC   |   LUX pass   |   Node D4   |   31 / 77   |   elapsed 04:12
```

Body, left to right:

**Grid widget (large).** 11 x 7 cells drawn to scale, labelled A1 through K7.
Cell colours: pending (grey), captured (green, opacity scaled to value so the map
builds up visually as the pass runs), current (blue outline, thick), flagged
high 3σ (amber), superseded (green with a small marker), reference node (blue
dot in the corner of the cell). Hovering a captured cell shows its mean, stdev
and 3σ%. Clicking any cell jumps the cursor there for a re-capture.

**Capture panel (right).**
- Huge live value readout for the active quantity only.
- **Capture** button, primary style, bound to **Space** and to **Enter**. Bind a
  second key (**F**) as an alias so a USB foot pedal that emits a keystroke can
  drive it hands-free.
- 3 s progress bar with countdown during a capture; the button is disabled and
  the grid cell pulses while sampling.
- Result card after each capture: `mean ± stdev  (3σ/avg X.X %, n=NN)`, coloured
  by the 3σ threshold.
- Secondary buttons: **Redo last**, **Skip node**, **Pause pass**,
  **End pass early**.

**Status bar.** Reuses `set_status()` semantics and the info/success/warning/error
styling from the measurement app.

### 7.4 Flow control

- **Variac prompt.** At the start of every level, a modal asks for the VAC value.
  Numeric validation, no default carried over from the previous level, so a stale
  value cannot be committed by reflex.
- **Warm-up.** After the VAC is entered, a countdown blocks captures. The
  timeseries plot stays live so the operator can see stability. A **Skip warm-up**
  button exists and, when used, writes `warmup_skipped=1` into the level record.
- **Reference captures** are forced, not skippable. The app jumps the cursor to
  the reference node and will not release it until the capture completes.
- **End of pass.** Shows drift %, flags it if over threshold, offers **Redo this
  pass** or **Continue**.
- **End of level.** Prompts: **Add another level** or **Finish campaign**.
- **Finish** runs the analysis and opens the Analysis tab.

### 7.5 Analysis tab

- Level and quantity selector.
- Heatmap of the selected map with node labels, a colorbar, the panel footprint
  drawn as a rectangle, and the sensor node marked. Raw / normalised toggle.
- Percent-deviation view: each node as % deviation from the panel-zone mean.
- Correction factor table, all levels, both quantities, with drift and uniformity
  columns.
- Level-independence readout: recommended K, spread %, and a plain-language
  verdict.
- Lux/irr ratio map.
- **Export** writes `analysis/` (all PNGs plus `correction_factors.csv`) and
  opens the folder with `open_in_explorer()`.

---

## 8. Testing

### 8.1 Simulation mode

`python src/lightbox_calibration.py --simulate` replaces both sensor threads with
a synthetic field generator: a plausible cosine-falloff profile centred on the
plate, scaled by the entered VAC, with ~1 % Gaussian noise and a slow 2 % linear
drift over a pass. This lets the whole GUI, the CSV writer, the resume path and
the analysis be exercised with no hardware and no lamp hours. The repo already
has precedent for this in the `E2E_TEST_PANEL` run names.

### 8.2 Unit tests (`tests/test_calibration_model.py`, no Qt, no hardware)

1. Serpentine ordering: 77 entries, no repeats, row 1 left to right, row 2 right
   to left, first node A1.
2. Label and coordinate round-trip: `F4` to (5, 3) to (250, 150) and back.
3. Panel footprint selection for several W x H values, including the
   fewer-than-4-nodes warning path.
4. Drift normalisation: with a synthetic 5 % linear drift, normalised values
   recover the flat field to within 0.1 %.
5. K computation against a hand-worked example.
6. Level-independence: constant-shape field at three levels gives
   `K_spread_pct < 0.1`.
7. CSV round-trip: write, read, filter `superseded`, same values out.
8. Resume: truncate `captures.csv` mid-pass, confirm the cursor restores to the
   correct node and the pass completes.

### 8.3 Hardware acceptance check (this is the one that matters)

The spatial map is only believable if something independent agrees with it. The
panel's own Isc is a true area integrator of irradiance.

1. Pick one panel and one variac level.
2. Sweep it at the centre position. Record Isc.
3. Shift it to a measurably different part of the map (a corner-ward offset of
   two or three nodes). Sweep again. Record Isc.
4. `Isc_centre / Isc_offset` should match
   `mean(irr over centre footprint) / mean(irr over offset footprint)` to within
   a few percent.

If those two ratios disagree, the map is wrong or the panel is not where you
think it is, and no amount of K arithmetic will fix it. Run this before trusting
the correction factor in any report.

---

## 9. Known limitations and blind spots

These are physics constraints on what the calibration can claim. Put them in the
app's About box and in any report that quotes a K value.

**Halogen spectrum shifts with variac voltage.** A halogen lamp run at reduced
voltage has a lower filament temperature and a redder spectrum. The lux meter is
photopic-weighted and the irradiance sensor is broadband, so the lux-per-W/m2
relationship changes level by level. Consequences: a lux-to-irradiance conversion
factor derived at one variac setting is invalid at another, and the lux map and
the irradiance map may have slightly different spatial shapes. This is precisely
why both quantities are captured at every node and every level, and why section
6.5 plots their ratio.

**Neither sensor sees the solar spectrum.** This calibration corrects for
position within the box. It says nothing about how box output relates to AM1.5.
Cross-panel comparisons under the same lamps stay valid; absolute outdoor
performance does not follow.

**Two passes means two lamp thermal states.** Bulb output drifts for several
minutes as the envelope heats. The warm-up countdown and the start/end reference
captures exist to bound this, and the reported drift % is the evidence. Always
run the LUX and IRR passes back to back at a level.

**Sensor head height.** The two heads probably sit at different heights above the
acrylic. With lamps this close, a few mm changes the reading. Record
`sensor_head_height_mm` per campaign and keep the jig geometry fixed between the
calibration and the real runs it corrects.

**K assumes the panel is where the covered nodes are.** Mark the panel outline on
the acrylic in permanent marker before the first real run. A panel placed 25 mm
off ruins the correction quietly.

**Grid pitch limits resolution.** A 50 mm pitch cannot see structure smaller than
about 100 mm. Individual lamp hotspots directly under the diffuser may be
narrower than that and will be aliased. If the map looks suspiciously smooth
directly over the lamp row, that is why.

---

## 10. Suggested build order

| Phase | Work | Output |
|---|---|---|
| 0 | Extract `src/sensors.py` from `low_light_app.py`, confirm the measurement app still runs unchanged | clean import surface |
| 1 | `calibration_model.py` plus its unit tests, no GUI | maths proven headlessly |
| 2 | GUI shell: Setup tab, Sensors tab, grid widget, `--simulate` mode | full flow exercised with no hardware |
| 3 | Run tab: capture loop, CSV writer, resume path | real captures possible |
| 4 | Analysis tab and export | correction factors produced |
| 5 | Hardware acceptance check per section 8.3 | map validated against panel Isc |

---

## 11. Practical note on session length

77 nodes x 2 passes is 154 captures per level. At roughly 8 s per capture
including the jig move, that is about 21 minutes of captures per level, call it
30 minutes with variac changes, warm-up and handling.

Before committing to a long campaign, run **three levels** (low, mid and high,
for example 20 / 60 / 110 VAC) and check the level-independence readout. If K is
constant to within a couple of percent across that range, the full sweep buys you
very little and a three-level campaign is the whole job. If K moves, the extra
levels are earning their keep and it is worth the afternoon.
