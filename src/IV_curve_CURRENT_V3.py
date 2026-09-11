#!/usr/bin/env python
import argparse
import atexit
import concurrent.futures
import csv
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
import uuid

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from otii_tcp_client import otii_connection, otii


HOST = {"IP": "127.0.0.1", "PORT": 1905}
SETTING = {"max_sink_current": 500e6, "interval": 105.333e-3}

# Adaptive sweep pacing (harvest()): the inter-step delay starts at
# SETTING["interval"] (the historically-known-safe pace) and adjusts from
# there based on how long each step's own Otii round trip actually takes --
# AIMD-style, like TCP congestion control: back off quickly (multiplicative)
# the moment Otii looks like it's struggling to keep up, speed up slowly
# (a few % at a time) while it's comfortably keeping pace. This is what
# protects against the hang the operator observed when commands are sent
# too fast for Otii to keep up with, while also letting most of the sweep
# run faster than the old fixed 105ms/step once Otii proves it can keep up.
ADAPTIVE_INTERVAL_MIN_S = 0.02
ADAPTIVE_INTERVAL_MAX_S = 0.5
ADAPTIVE_SPEEDUP_FACTOR = 0.95
ADAPTIVE_BACKOFF_FACTOR = 2.0
ADAPTIVE_FAST_RATIO = 0.4
ADAPTIVE_SLOW_RATIO = 0.8

proj = None
Isc = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run IV measurement and append lux/irradiance/IV metrics to summary CSV"
    )
    parser.add_argument(
        "--summary-csv",
        default=str(Path(__file__).resolve().parents[1] / "report" / "LowLightTesting_summary.csv"),
        help="Path to append-only summary CSV",
    )
    parser.add_argument("--sensor-host", default="127.0.0.1", help="listen.py TCP host")
    parser.add_argument("--sensor-port", type=int, default=8765, help="listen.py TCP port")
    parser.add_argument("--sensor-timeout", type=float, default=6.0, help="Snapshot socket timeout (s)")
    parser.add_argument(
        "--iv-timeout-seconds",
        type=float,
        default=50.0,
        help="Timeout for IV sweep before recovery (default: 50s)",
    )
    parser.add_argument(
        "--otii-exe",
        default=r"C:\Users\aniru\AppData\Local\otii3\Otii 3.exe",
        help="Path to Otii 3 executable",
    )
    parser.add_argument(
        "--otii-restart-wait-seconds",
        type=float,
        default=5.0,
        help="Seconds to wait after restarting Otii app",
    )
    parser.add_argument("--campaign", action="store_true", help="Run operator-confirmed continuous campaign loop")
    parser.add_argument(
        "--state-json",
        default=str(Path(__file__).resolve().parents[1] / "report" / "iv_campaign_state.json"),
        help="Campaign state file for resume/progress",
    )
    parser.add_argument(
        "--show-plot",
        dest="show_plot",
        action="store_true",
        default=True,
        help="Show IV plot window (default: on)",
    )
    parser.add_argument(
        "--no-show-plot",
        dest="show_plot",
        action="store_false",
        help="Disable IV plot window",
    )
    return parser.parse_args()


def cleanup(my_arcs):
    try:
        if proj is not None:
            proj.stop_recording()
    except Exception:
        pass
    for my_arc in my_arcs:
        my_arc.set_main_current(0)
        my_arc.set_main(False)
        my_arc.set_power_regulation("voltage")


def iv_get_inputs():
    root = tk.Tk()
    root.title("Input Fields")

    default_values = {
        "Panel ID": "P393R2A",
        "Light Meter": "Sep21",
        "Solar Intensity": "1000",
    }

    panel_id_label = tk.Label(root, text="Panel ID:")
    panel_id_label.pack()
    panel_id_entry = tk.Entry(root)
    panel_id_entry.insert(0, default_values["Panel ID"])
    panel_id_entry.pack()

    light_meter_label = tk.Label(root, text="Light Meter:")
    light_meter_label.pack()
    light_meter_entry = tk.Entry(root)
    light_meter_entry.insert(0, default_values["Light Meter"])
    light_meter_entry.pack()

    solar_intensity_label = tk.Label(root, text="Solar Intensity:")
    solar_intensity_label.pack()
    solar_intensity_entry = tk.Entry(root)
    solar_intensity_entry.insert(0, default_values["Solar Intensity"])
    solar_intensity_entry.pack()

    inputs = {}

    def on_ok():
        inputs.update(
            {
                "Panel ID": panel_id_entry.get().strip(),
                "Light Meter": light_meter_entry.get().strip(),
                "Solar Intensity": solar_intensity_entry.get().strip(),
            }
        )
        root.destroy()

    ok_button = tk.Button(root, text="OK", command=on_ok)
    ok_button.pack()

    root.update_idletasks()
    window_width = min(420, root.winfo_screenwidth() - 100)
    window_height = min(240, root.winfo_screenheight() - 100)
    x_position = (root.winfo_screenwidth() - window_width) // 2
    y_position = (root.winfo_screenheight() - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    root.mainloop()
    return inputs


def check_create_project(otii_object):
    active_project = otii_object.get_active_project()
    if active_project:
        print("Project already active")
        return active_project
    print("Project created")
    return otii_object.create_project()


def connect_otii_session():
    connection = otii_connection.OtiiConnection(HOST["IP"], HOST["PORT"])
    connect_response = connection.connect_to_server()
    if connect_response["type"] == "error":
        raise RuntimeError(
            "Exit! Error code: "
            + connect_response["errorcode"]
            + ", Description: "
            + connect_response["data"]["message"]
        )

    otii_object = otii.Otii(connection)
    devices = otii_object.get_devices()
    if not devices:
        raise RuntimeError("Must have Arcs connected!")

    active_proj = check_create_project(otii_object)
    return connection, otii_object, active_proj, devices


def restart_otii_app(otii_exe, wait_seconds):
    print("Restarting Otii 3 app (force kill)...")
    subprocess.run(
        ["taskkill", "/IM", "Otii 3.exe", "/F"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["taskkill", "/IM", "Otii3.exe", "/F"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    exe_path = Path(otii_exe)
    if not exe_path.exists():
        raise RuntimeError(f"Otii executable not found: {exe_path}")

    subprocess.Popen([str(exe_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(max(wait_seconds, 0.0))


def recover_otii_connection(args):
    try:
        restart_otii_app(args.otii_exe, args.otii_restart_wait_seconds)
    except Exception as e:
        print(f"Automatic Otii restart failed: {e}")

    while True:
        input("Reconnect Arc device and ensure Otii is open, then press Enter to continue... ")
        try:
            _, _, active_proj, devices = connect_otii_session()
            print("Reconnected to Otii and detected Arc device(s).")
            return active_proj, devices
        except Exception as e:
            print(f"Reconnect failed: {e}")
            choice = input("Retry reconnect (r) or quit campaign (q): ").strip().lower()
            if choice == "q":
                raise RuntimeError("Operator aborted reconnect after timeout")


def _call_with_thread_timeout(fn, timeout_s, *args, **kwargs):
    """Run fn(*args, **kwargs) with a hard timeout. start_recording()/
    stop_recording() have no timeout of their own; if one hangs, the only
    thing that used to notice was the *outer* call_with_timeout() wrapping
    the whole harvest()/short_circuit() call from low_light_app.py -- which
    meant the operator could see the sweep's progress bar reach 100% (the
    sweep loop itself had already finished) while the measurement then sat
    unresponsive for the entire remaining timeout budget before the app's
    comms-lost recovery finally kicked in (2026-09 feedback: "IV curve
    fails to terminate even though the progress bar shows 100%"). This
    surfaces that specific hang in `timeout_s` seconds instead.
    Like asyncio.to_thread, this cannot actually kill a genuinely stuck
    call -- if it times out, the underlying thread is simply abandoned
    (the caller regains control immediately; the orphaned thread exits on
    its own once/if the blocking call ever returns)."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return executor.submit(fn, *args, **kwargs).result(timeout=timeout_s)
    finally:
        executor.shutdown(wait=False)


def short_circuit(active_proj, my_arcs):
    global Isc
    atexit.register(cleanup, my_arcs)
    for my_arc in my_arcs:
        my_arc.set_main_current(0)
        my_arc.set_power_regulation("inline")
        my_arc.enable_channel("mv", True)
        my_arc.enable_channel("mc", True)
        my_arc.set_main(True)
    time.sleep(0.1)

    _call_with_thread_timeout(active_proj.start_recording, 10.0)
    time.sleep(0.5)
    recording = active_proj.get_last_recording()
    Isc = -1 * my_arcs[0].get_value("mc")
    _call_with_thread_timeout(active_proj.stop_recording, 10.0)
    time.sleep(0.2)
    return float(Isc)


def harvest(
    active_proj,
    my_arcs,
    current_step,
    timeout_seconds=50.0,
    progress_cb=None,
    live_data_cb=None,
    live_data_interval_s=1.0,
    expected_max_current_uA=None,
    stop_event=None,
):
    if expected_max_current_uA is None or expected_max_current_uA <= 0:
        # Callers derive current_step from Isc as current_step = Isc/150
        # (see run_single_measurement()/_async_start_measurement()), so a
        # full sweep is ~150 steps -- i.e. current_uA naturally tops out
        # around current_step*150, NOT SETTING["max_sink_current"] (500A).
        # The sweep always terminates at the voltage cutoff, far below
        # 500A, so using that as the progress denominator made the
        # progress bar sit at ~0% for the entire sweep and only jump to
        # 100% at the very end (2026-09 feedback: "progress bar not
        # working").
        expected_max_current_uA = current_step * 150.0
    atexit.register(cleanup, my_arcs)
    for my_arc in my_arcs:
        my_arc.set_main_current(0)
        my_arc.set_power_regulation("current")
        my_arc.enable_channel("mv", True)
        my_arc.enable_channel("mc", True)
        my_arc.enable_channel("mp", True)
        my_arc.set_channel_samplerate("mv", 1000)
        my_arc.set_channel_samplerate("mc", 1000)
        my_arc.set_channel_samplerate("mp", 1000)
        my_arc.set_main(True)
    time.sleep(0.1)

    recording_started = False
    recording = None
    try:
        _call_with_thread_timeout(active_proj.start_recording, 10.0)
        recording_started = True
        time.sleep(0.5)
        recording = active_proj.get_last_recording()

        ready = []
        started = time.monotonic()
        last_live_fetch = 0.0
        interval = SETTING["interval"]
        for current_uA in range(0, int(SETTING["max_sink_current"]), int(current_step)):
            now = time.monotonic()
            if (now - started) > timeout_seconds:
                raise TimeoutError(f"IV measurement timeout after {timeout_seconds:.1f}s")

            if stop_event is not None and stop_event.is_set():
                # Operator-requested early stop ("End Sweep Now") -- wind
                # down any arc that hasn't already hit the voltage cutoff,
                # then finish exactly like a normal completion. Whatever
                # the recording captured up to this point is treated as
                # the final, valid data (explicit operator instruction).
                for my_arc in my_arcs:
                    if my_arc.id not in ready:
                        my_arc.set_main(False)
                        my_arc.set_main_current(0)
                        my_arc.set_power_regulation("voltage")
                break

            if progress_cb is not None:
                progress_cb(min(1.0, current_uA / expected_max_current_uA))

            if live_data_cb is not None and (now - last_live_fetch) >= live_data_interval_s:
                last_live_fetch = now
                try:
                    # Best-effort only: runs on this same thread/connection,
                    # never a separate one, so it can't race the sweep's own
                    # requests -- but a live-preview hiccup must still never
                    # break the actual measurement.
                    sample_count = recording.get_channel_data_count(my_arcs[0].id, "mv")
                    if sample_count > 0:
                        mv_live = recording.get_channel_data(my_arcs[0].id, "mv", 0, sample_count)["values"]
                        mc_live = recording.get_channel_data(my_arcs[0].id, "mc", 0, sample_count)["values"]
                        live_data_cb(mv_live, mc_live)
                except Exception:
                    pass

            step_started = time.monotonic()
            for my_arc in my_arcs:
                my_arc.set_main_current(-current_uA * 1e-6)
                if my_arc.get_value("mv") < 0.2:
                    my_arc.set_main(False)
                    my_arc.set_main_current(0)
                    my_arc.set_power_regulation("voltage")
                    ready.append(my_arc.id)
            call_latency = time.monotonic() - step_started

            if len(ready) >= len(my_arcs):
                break

            if call_latency > ADAPTIVE_SLOW_RATIO * interval:
                # Otii's own round trip is eating most (or more) of our
                # pacing budget -- back off decisively rather than risk
                # piling commands up faster than it can process them.
                interval = min(ADAPTIVE_INTERVAL_MAX_S, max(interval * ADAPTIVE_BACKOFF_FACTOR, call_latency * 1.5))
            elif call_latency < ADAPTIVE_FAST_RATIO * interval:
                # Comfortably keeping up -- ease the pace up a little.
                interval = max(ADAPTIVE_INTERVAL_MIN_S, interval * ADAPTIVE_SPEEDUP_FACTOR)

            time.sleep(interval)
    finally:
        if recording_started:
            try:
                _call_with_thread_timeout(active_proj.stop_recording, 10.0)
            except Exception:
                pass
        time.sleep(0.2)

    # Only reached once cleanup above has actually finished -- previously
    # progress_cb(1.0) fired (and the GUI's progress bar showed 100%)
    # *before* stop_recording() ran, so if that call hung, the operator saw
    # a "complete" progress bar while the measurement was actually still
    # stuck for up to the entire remaining timeout budget (2026-09
    # feedback). Now the bar only reaches 100% once harvest() is truly done.
    if progress_cb is not None:
        progress_cb(1.0)

    return recording


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def request_sensor_snapshot(host, port, timeout_s):
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall(b"SNAPSHOT\n")
        payload = sock.recv(16384)

    if not payload:
        raise RuntimeError("No response from listen.py SNAPSHOT request")

    text = payload.decode("utf-8", errors="ignore").strip()
    try:
        snapshot = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid SNAPSHOT JSON response: {text}") from e

    if not snapshot.get("ok"):
        raise RuntimeError(f"listen.py snapshot failed: {snapshot}")

    if "lux" not in snapshot or "irradiance" not in snapshot:
        raise RuntimeError(f"listen.py snapshot missing lux/irradiance fields: {snapshot}")

    return snapshot


def copy_to_clipboard(text):
    try:
        import pyperclip

        pyperclip.copy(text)
        return True
    except Exception:
        pass

    try:
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
        return True
    except Exception:
        return False


SUMMARY_CSV_FIELDNAMES = [
    "session_id",
    "run_index",
    "status",
    "timestamp",
    "panel_type",
    "panel_name",
    "light_meter",
    "irradiance_gui_input",
    "lux_measured",
    "irradiance_measured_live",
    "irradiance_live_samples",
    "Wp",
    "Vp",
    "Ip",
    "Voc",
    "Isc",
    "notes",
    "error",
    "source",
    "panel_temp_c",
    "lux_stdev",
    "irradiance_stdev",
    "lux_3sigma_pct",
    "irradiance_3sigma_pct",
    "png_path",
    "csv_path",
]


def append_summary_csv(csv_path, row_values):
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_file.exists()
    with csv_file.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row_values.get(k, "") for k in SUMMARY_CSV_FIELDNAMES})

    return csv_file


def rewrite_summary_csv(csv_path, rows):
    """Rewrite the whole summary CSV from a list of row dicts (the same
    shape make_row() produces) -- used when an already-saved row is edited
    in place (e.g. renaming a panel in the GUI's session table), since
    append_summary_csv() only ever adds new rows. Overwrites the file
    entirely; the caller is responsible for `rows` being the complete,
    current set (GUI: self.session_rows, which mirrors the table)."""
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    with csv_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_CSV_FIELDNAMES)
        writer.writeheader()
        for row_values in rows:
            writer.writerow({k: row_values.get(k, "") for k in SUMMARY_CSV_FIELDNAMES})

    return csv_file


def read_max_run_index(csv_path):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        return 0

    max_idx = 0
    with csv_file.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                idx = int(str(row.get("run_index", "")).strip())
            except Exception:
                continue
            if idx > max_idx:
                max_idx = idx
    return max_idx


def load_campaign_state(state_path):
    path = Path(state_path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_campaign_state(state_path, state):
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def fetch_recording_channels(recording, my_arc):
    """The network-bound half of plot_iv_curve(): pulls the mv/mc/mp channel
    data for a completed recording from Otii. Split out so a caller that
    needs to keep this off a UI thread (it's an unbounded Otii TCP
    round-trip, same hang risk as start_recording/stop_recording) can run it
    via its own timeout/threading and hand the result to render_iv_curve()
    -- see low_light_app.py's _async_start_measurement()."""
    mv_samples = recording.get_channel_data_count(my_arc.id, "mv")
    mv_data_dict = recording.get_channel_data(my_arc.id, "mv", 0, mv_samples)
    mc_data_dict = recording.get_channel_data(my_arc.id, "mc", 0, mv_samples)
    mp_data_dict = recording.get_channel_data(my_arc.id, "mp", 0, mv_samples)

    if "values" not in mv_data_dict or "values" not in mc_data_dict:
        raise RuntimeError("Data retrieval failed. Please check the recording and connection.")

    return mv_data_dict, mc_data_dict, mp_data_dict


def render_iv_curve(
    mv_data_dict, mc_data_dict, mp_data_dict, panel_id, solar_intensity, light_meter,
    show_plot=True, fig=None, out_dir=None,
):
    """The local (no network) half of plot_iv_curve(): computes metrics,
    plots, and saves the PNG/CSV from already-fetched channel data. Pure
    number-crunching + file I/O -- safe to run wherever the caller likes
    (this is the part low_light_app.py runs on the Qt main thread, since it
    touches the shared canvas Figure, once fetch_recording_channels() has
    already gotten the data off the network on another thread)."""
    mv_data = np.array(mv_data_dict["values"])
    mc_data = np.array(mc_data_dict["values"])
    mp_data = np.array(mp_data_dict["values"])

    if len(mv_data) != len(mc_data):
        raise RuntimeError("Voltage and current data have different lengths.")

    window_size = 25
    skip = 10
    mv_data = moving_average(mv_data, window_size)[:-skip]
    mc_data = -1 * moving_average(mc_data, window_size)[:-skip]
    power_data = -1 * moving_average(mp_data, window_size)[:-skip]

    max_power = float(np.max(power_data))
    max_index = int(np.argmax(power_data))
    max_voltage = float(mv_data[max_index])
    max_current = float(mc_data[max_index])
    voc = float(np.max(mv_data))
    isc = float(np.max(mc_data))
    ff = 0.0 if (isc * voc) == 0 else (max_power / (isc * voc))

    owns_fig = fig is None
    if owns_fig:
        fig, ax1 = plt.subplots(figsize=(1680 / 100, 920 / 100))
    else:
        fig.clf()
        ax1 = fig.add_subplot(111)

    ax1.set_xlabel("Voltage (V)")
    ax1.set_ylabel("Current (uA)", color="blue")
    ax1.plot(mv_data, 1e6 * mc_data, color="blue", label="Current (uA)")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Power (uW)", color="red")
    ax2.plot(mv_data, 1e6 * power_data, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    ax2.scatter(max_voltage, 1e6 * max_power, color="red", marker="o", label="Max Power Point")
    ax2.annotate(
        f"Vmpp={max_voltage:.2f} V, \nImpp={1e6 * max_current:.2f} uA",
        xy=(max_voltage, 1e6 * max_power),
        xytext=(max_voltage + 0.5, 1e6 * max_power + 1),
        arrowprops={"arrowstyle": "->", "linestyle": "dotted"},
    )

    ax1.grid()
    ax2.legend(loc="upper center")
    ax1.yaxis.label.set_color("blue")
    ax2.yaxis.label.set_color("red")
    ax1.set_title(f"I-V Curve: {panel_id} at {solar_intensity} W/m2\nLightmeter Cal.: {light_meter}")

    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    text_x = xlim[0] + (1 / 16) * (xlim[1] - xlim[0])
    text_y = ylim[1] - (3 / 9) * (ylim[1] - ylim[0])
    ax1.text(
        text_x,
        text_y,
        (
            "Panel Performance Parameters:\n"
            f"Isc: {1e6 * isc:.2f} uA\n"
            f"Impp: {1e6 * max_current:.2f} uA\n"
            f"Voc: {voc:.2f} V\n"
            f"Vmpp: {max_voltage:.2f} V\n"
            f"PMax: {1e6 * max_power:.2f} uW\n"
            f"FF: {ff:.2%}"
        ),
        bbox={"facecolor": "white", "alpha": 0.7},
    )

    timestamp = time.strftime("%Y-%m-%d-%H-%M")
    out_path = Path(out_dir) if out_dir else Path(".")
    out_path.mkdir(parents=True, exist_ok=True)
    png_path = out_path / f"{timestamp} Panel {panel_id} SI {solar_intensity}.png"
    csv_path = out_path / f"{timestamp} Panel {panel_id} SI {solar_intensity}.csv"
    fig.savefig(png_path)
    np.savetxt(
        csv_path,
        np.column_stack((mv_data, mc_data, power_data)),
        delimiter=",",
        header="Voltage (V), Current (A), Power (W)",
        comments="",
    )
    if owns_fig:
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    print(f"i_sc: {1e6 * isc:.2f} uA")
    print(f"i_mpp: {1e6 * max_current:.2f} uA")
    print(f"v_oc: {voc:.2f} V")
    print(f"v_mpp: {max_voltage:.2f} V")
    print(f"p_mpp: {1e6 * max_power:.2f} uW")
    print(f"ff: {ff:.2%}")

    return {
        "Wp": max_power,
        "Vp": max_voltage,
        "Ip": max_current,
        "Voc": voc,
        "Isc": isc,
        # Recorded so a later panel-name edit (GUI session table) can find
        # and rename these exact files instead of guessing a filename.
        "png_path": str(png_path),
        "csv_path": str(csv_path),
    }


def plot_iv_curve(recording, my_arc, panel_id, solar_intensity, light_meter, show_plot=True, fig=None, out_dir=None):
    """Convenience wrapper combining fetch_recording_channels() +
    render_iv_curve() for callers (the CLI campaign flow) that don't need
    the network fetch off any particular thread."""
    mv_data_dict, mc_data_dict, mp_data_dict = fetch_recording_channels(recording, my_arc)
    return render_iv_curve(
        mv_data_dict, mc_data_dict, mp_data_dict, panel_id, solar_intensity, light_meter,
        show_plot=show_plot, fig=fig, out_dir=out_dir,
    )


def derive_panel_type(panel_name):
    return panel_name.split("_", 1)[0] if "_" in panel_name else panel_name


def _fmt_float(value, precision):
    if value in (None, ""):
        return ""
    try:
        return format(float(value), precision)
    except Exception:
        return str(value)


def build_clipboard_row(row_values):
    ordered = [
        row_values.get("session_id", ""),
        row_values.get("run_index", ""),
        row_values.get("status", ""),
        row_values.get("timestamp", ""),
        row_values.get("panel_type", ""),
        row_values.get("panel_name", ""),
        row_values.get("light_meter", ""),
        row_values.get("irradiance_gui_input", ""),
        _fmt_float(row_values.get("lux_measured"), ".3f"),
        _fmt_float(row_values.get("irradiance_measured_live"), ".6f"),
        row_values.get("irradiance_live_samples", ""),
        _fmt_float(row_values.get("Wp"), ".6g"),
        _fmt_float(row_values.get("Vp"), ".6g"),
        _fmt_float(row_values.get("Ip"), ".6g"),
        _fmt_float(row_values.get("Voc"), ".6g"),
        _fmt_float(row_values.get("Isc"), ".6g"),
        row_values.get("notes", ""),
        row_values.get("error", ""),
    ]
    return "\t".join(str(v) for v in ordered) + "\n"


def prompt_with_default(label, default=""):
    prompt = f"{label}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "
    value = input(prompt).strip()
    return value if value else default


def prompt_required(label):
    while True:
        value = input(f"{label}: ").strip()
        if value:
            return value
        print(f"{label} is required.")


def prompt_campaign_entry(last_light_meter):
    while True:
        panel_name = prompt_required("Panel name")
        gui_irradiance_input = prompt_required("Irradiance (other sensor)")
        light_meter = prompt_with_default("Light meter", last_light_meter)
        notes = input("Notes (optional): ").strip()

        print("\nConfirm run input:")
        print(f"  panel_name: {panel_name}")
        print(f"  irradiance_gui_input: {gui_irradiance_input}")
        print(f"  light_meter: {light_meter}")
        print(f"  notes: {notes}")

        action = input("Enter=run, e=edit, s=skip, q=quit: ").strip().lower()
        if action == "q":
            return None, "quit"
        if action == "s":
            return {
                "panel_name": panel_name,
                "gui_irradiance_input": gui_irradiance_input,
                "light_meter": light_meter,
                "notes": notes,
            }, "skip"
        if action == "e":
            continue
        return {
            "panel_name": panel_name,
            "gui_irradiance_input": gui_irradiance_input,
            "light_meter": light_meter,
            "notes": notes,
        }, "run"


def run_single_measurement(args, active_proj, devices, panel_name, gui_irradiance_input, light_meter):
    isc_measured = short_circuit(active_proj, devices)
    if isc_measured <= 0:
        raise RuntimeError(f"Invalid Isc measured: {isc_measured}")

    current_step = (isc_measured / 150) * 1e6
    if current_step <= 0:
        raise RuntimeError(f"Invalid current step derived from Isc: {current_step}")

    recording = harvest(active_proj, devices, current_step, timeout_seconds=args.iv_timeout_seconds)
    my_arc = devices[0]
    metrics = plot_iv_curve(
        recording,
        my_arc,
        panel_name,
        gui_irradiance_input,
        light_meter,
        show_plot=args.show_plot,
    )
    snapshot = request_sensor_snapshot(args.sensor_host, args.sensor_port, args.sensor_timeout)
    return metrics, snapshot


def make_row(
    session_id,
    run_index,
    status,
    panel_name,
    light_meter,
    gui_irradiance_input,
    notes,
    metrics=None,
    snapshot=None,
    error="",
    panel_temp_c="",
    lux_stdev="",
    irradiance_stdev="",
    lux_3sigma_pct="",
    irradiance_3sigma_pct="",
    png_path="",
    csv_path="",
):
    row = {
        "session_id": session_id,
        "run_index": run_index,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "panel_type": derive_panel_type(panel_name),
        "panel_name": panel_name,
        "light_meter": light_meter,
        "irradiance_gui_input": gui_irradiance_input,
        "lux_measured": "",
        "irradiance_measured_live": "",
        "irradiance_live_samples": "",
        "Wp": "",
        "Vp": "",
        "Ip": "",
        "Voc": "",
        "Isc": "",
        "notes": notes,
        "error": error,
        "panel_temp_c": panel_temp_c,
        "lux_stdev": lux_stdev,
        "irradiance_stdev": irradiance_stdev,
        "lux_3sigma_pct": lux_3sigma_pct,
        "irradiance_3sigma_pct": irradiance_3sigma_pct,
        "png_path": png_path,
        "csv_path": csv_path,
        "source": "iv_curve_current_v3",
    }

    if snapshot is not None:
        row["timestamp"] = snapshot.get("ts", row["timestamp"])
        row["lux_measured"] = float(snapshot.get("lux", 0.0))
        row["irradiance_measured_live"] = float(snapshot.get("irradiance", 0.0))
        row["irradiance_live_samples"] = int(snapshot.get("samples", 0))

    if metrics is not None:
        row["Wp"] = float(metrics["Wp"])
        row["Vp"] = float(metrics["Vp"])
        row["Ip"] = float(metrics["Ip"])
        row["Voc"] = float(metrics["Voc"])
        row["Isc"] = float(metrics["Isc"])
        # render_iv_curve()/plot_iv_curve() already record the exact files
        # they wrote as part of metrics -- fall back to those if the caller
        # didn't pass png_path/csv_path explicitly.
        if not row["png_path"]:
            row["png_path"] = metrics.get("png_path", "")
        if not row["csv_path"]:
            row["csv_path"] = metrics.get("csv_path", "")

    return row


def run_campaign(args, active_proj, devices):
    global proj
    proj = active_proj
    campaign_start = time.time()
    state = load_campaign_state(args.state_json)
    max_csv_index = read_max_run_index(args.summary_csv)

    session_id = state.get("session_id") or f"campaign-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    next_run_index = max(int(state.get("next_run_index", 1)), max_csv_index + 1)
    runs = state.get("runs", [])
    last_light_meter = state.get("last_light_meter", "")

    state = {
        "session_id": session_id,
        "next_run_index": next_run_index,
        "last_light_meter": last_light_meter,
        "runs": runs,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_campaign_state(args.state_json, state)

    print(f"Campaign session: {session_id}")
    print(f"Starting run_index: {next_run_index}")

    counts = {"done": 0, "failed": 0, "skipped": 0}

    def print_summary():
        elapsed = time.time() - campaign_start
        total = counts["done"] + counts["failed"] + counts["skipped"]
        per_hour = 0.0 if elapsed <= 0 else (counts["done"] / elapsed) * 3600.0
        print("\nCampaign summary")
        print(f"  session_id: {session_id}")
        print(f"  total processed: {total}")
        print(f"  done: {counts['done']}, failed: {counts['failed']}, skipped: {counts['skipped']}")
        print(f"  elapsed: {elapsed/60.0:.1f} min")
        print(f"  throughput: {per_hour:.1f} successful runs/hour")

    while True:
        print("\n--- Next IV Run ---")
        entry, action = prompt_campaign_entry(state.get("last_light_meter", ""))

        if action == "quit":
            print("Campaign stopped by operator.")
            print_summary()
            break

        if entry is None:
            print("No entry provided. Stopping campaign.")
            print_summary()
            break

        run_index = int(state["next_run_index"])
        state["last_light_meter"] = entry["light_meter"]

        if action == "skip":
            row = make_row(
                session_id=session_id,
                run_index=run_index,
                status="skipped",
                panel_name=entry["panel_name"],
                light_meter=entry["light_meter"],
                gui_irradiance_input=entry["gui_irradiance_input"],
                notes=entry["notes"],
                error="operator_skip",
            )
            append_summary_csv(args.summary_csv, row)
            copy_to_clipboard(build_clipboard_row(row))
            counts["skipped"] += 1
            state["runs"].append({"run_index": run_index, "status": "skipped", "panel_name": entry["panel_name"]})
            state["next_run_index"] = run_index + 1
            state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            save_campaign_state(args.state_json, state)
            print(f"Run {run_index}: skipped")
            continue

        auto_retried_after_timeout = False
        while True:
            try:
                metrics, snapshot = run_single_measurement(
                    args,
                    active_proj,
                    devices,
                    entry["panel_name"],
                    entry["gui_irradiance_input"],
                    entry["light_meter"],
                )
                row = make_row(
                    session_id=session_id,
                    run_index=run_index,
                    status="done",
                    panel_name=entry["panel_name"],
                    light_meter=entry["light_meter"],
                    gui_irradiance_input=entry["gui_irradiance_input"],
                    notes=entry["notes"],
                    metrics=metrics,
                    snapshot=snapshot,
                )
                append_summary_csv(args.summary_csv, row)
                copy_to_clipboard(build_clipboard_row(row))
                counts["done"] += 1
                state["runs"].append(
                    {
                        "run_index": run_index,
                        "status": "done",
                        "panel_name": entry["panel_name"],
                        "timestamp": row["timestamp"],
                    }
                )
                state["next_run_index"] = run_index + 1
                state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                save_campaign_state(args.state_json, state)
                print(
                    f"Run {run_index}: done | panel={entry['panel_name']} | "
                    f"lux={_fmt_float(row['lux_measured'], '.3f')} | "
                    f"irr_live={_fmt_float(row['irradiance_measured_live'], '.6f')} | "
                    f"Wp={_fmt_float(row['Wp'], '.6g')}"
                )
                break
            except TimeoutError as e:
                print(f"Run {run_index} timed out: {e}")
                try:
                    active_proj, devices = recover_otii_connection(args)
                    proj = active_proj
                except Exception as reconnect_error:
                    print(f"Otii recovery failed: {reconnect_error}")
                    choice = input("Skip run (s) or quit campaign (q): ").strip().lower()
                    if choice == "s":
                        row = make_row(
                            session_id=session_id,
                            run_index=run_index,
                            status="failed",
                            panel_name=entry["panel_name"],
                            light_meter=entry["light_meter"],
                            gui_irradiance_input=entry["gui_irradiance_input"],
                            notes=entry["notes"],
                            error=f"iv_timeout_recovery_failed: {e}",
                        )
                        append_summary_csv(args.summary_csv, row)
                        copy_to_clipboard(build_clipboard_row(row))
                        counts["failed"] += 1
                        state["runs"].append(
                            {
                                "run_index": run_index,
                                "status": "failed",
                                "panel_name": entry["panel_name"],
                                "error": f"iv_timeout_recovery_failed: {e}",
                            }
                        )
                        state["next_run_index"] = run_index + 1
                        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                        save_campaign_state(args.state_json, state)
                        break

                    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    save_campaign_state(args.state_json, state)
                    print("Campaign stopped after timeout recovery failure.")
                    print_summary()
                    return

                if not auto_retried_after_timeout:
                    auto_retried_after_timeout = True
                    print("Otii recovered. Auto-retrying this run once...")
                    continue

                choice = input("Retry run (r), skip (s), or quit (q): ").strip().lower()
                if choice == "r":
                    continue
                if choice == "s":
                    row = make_row(
                        session_id=session_id,
                        run_index=run_index,
                        status="failed",
                        panel_name=entry["panel_name"],
                        light_meter=entry["light_meter"],
                        gui_irradiance_input=entry["gui_irradiance_input"],
                        notes=entry["notes"],
                        error=f"iv_timeout: {e}",
                    )
                    append_summary_csv(args.summary_csv, row)
                    copy_to_clipboard(build_clipboard_row(row))
                    counts["failed"] += 1
                    state["runs"].append(
                        {
                            "run_index": run_index,
                            "status": "failed",
                            "panel_name": entry["panel_name"],
                            "error": f"iv_timeout: {e}",
                        }
                    )
                    state["next_run_index"] = run_index + 1
                    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    save_campaign_state(args.state_json, state)
                    break

                state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                save_campaign_state(args.state_json, state)
                print("Campaign stopped after timeout.")
                print_summary()
                return
            except Exception as e:
                print(f"Run {run_index} failed: {e}")
                choice = input("Retry (r), skip (s), quit (q): ").strip().lower()
                if choice == "r":
                    continue
                if choice == "s":
                    row = make_row(
                        session_id=session_id,
                        run_index=run_index,
                        status="failed",
                        panel_name=entry["panel_name"],
                        light_meter=entry["light_meter"],
                        gui_irradiance_input=entry["gui_irradiance_input"],
                        notes=entry["notes"],
                        error=str(e),
                    )
                    append_summary_csv(args.summary_csv, row)
                    copy_to_clipboard(build_clipboard_row(row))
                    counts["failed"] += 1
                    state["runs"].append(
                        {
                            "run_index": run_index,
                            "status": "failed",
                            "panel_name": entry["panel_name"],
                            "error": str(e),
                        }
                    )
                    state["next_run_index"] = run_index + 1
                    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    save_campaign_state(args.state_json, state)
                    break
                if choice == "q":
                    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    save_campaign_state(args.state_json, state)
                    print("Campaign stopped after failure.")
                    print_summary()
                    return

    print_summary()


def main():
    global proj
    args = parse_args()

    try:
        _connection, _otii_object, active_proj, devices = connect_otii_session()
    except Exception as e:
        print(str(e))
        return 1

    proj = active_proj
    if args.campaign:
        run_campaign(args, proj, devices)
        print("Done!")
        return 0

    inputs = iv_get_inputs()
    panel_name = inputs.get("Panel ID", "").strip()
    if not panel_name:
        print("Panel ID is required.")
        return 2
    light_meter = inputs.get("Light Meter", "")
    gui_irradiance_input = inputs.get("Solar Intensity", "")

    try:
        metrics, snapshot = run_single_measurement(
            args,
            proj,
            devices,
            panel_name,
            gui_irradiance_input,
            light_meter,
        )
    except TimeoutError as e:
        print(f"Measurement timed out: {e}")
        try:
            proj, devices = recover_otii_connection(args)
            print("Retrying measurement once after Otii recovery...")
            metrics, snapshot = run_single_measurement(
                args,
                proj,
                devices,
                panel_name,
                gui_irradiance_input,
                light_meter,
            )
        except Exception as retry_err:
            print(f"Measurement failed after recovery: {retry_err}")
            return 1
    except Exception as e:
        print(f"Measurement failed: {e}")
        return 1

    run_index = read_max_run_index(args.summary_csv) + 1
    row_values = make_row(
        session_id="single",
        run_index=run_index,
        status="done",
        panel_name=panel_name,
        light_meter=light_meter,
        gui_irradiance_input=gui_irradiance_input,
        notes="",
        metrics=metrics,
        snapshot=snapshot,
    )

    clipboard_row = build_clipboard_row(row_values)
    copied = copy_to_clipboard(clipboard_row)
    if copied:
        print("Copied summary row to clipboard.")
    else:
        print("Warning: failed to copy row to clipboard.")

    try:
        csv_path = append_summary_csv(args.summary_csv, row_values)
    except Exception as e:
        print(f"Summary CSV write failed: {e}")
        print("Clipboard row prepared as fallback.")
        return 1

    print(f"Appended summary row in {csv_path}")
    print(
        "Saved values -> "
        f"run={row_values['run_index']}, panel={row_values['panel_name']}, "
        f"gui_irr={row_values['irradiance_gui_input']}, "
        f"lux={_fmt_float(row_values['lux_measured'], '.3f')}, "
        f"irr_live={_fmt_float(row_values['irradiance_measured_live'], '.6f')}, "
        f"Wp={_fmt_float(row_values['Wp'], '.6g')}, Vp={_fmt_float(row_values['Vp'], '.6g')}, "
        f"Ip={_fmt_float(row_values['Ip'], '.6g')}, Voc={_fmt_float(row_values['Voc'], '.6g')}, "
        f"Isc={_fmt_float(row_values['Isc'], '.6g')}"
    )

    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
