"""
Real-time gait event detection via UDP.

Receives HTC VIVE tracker pose packets (msgpack over UDP), applies causal
filtering, and runs the same gait event algorithm as the offline pipeline
using GaitEventDetector.detect_step() one sample at a time.

Architecture
------------
- UDP Listener thread : receives packets, runs detect_step(), updates shared cache
- Dash web app        : live plot of filtered Z signal + left/right foot state
- Tkinter Toolkit     : GUI controls for calibration reset and event export

Usage
-----
1. Run the offline pipeline (main.py) on a calibration trial to produce offsets.json.
2. Start this script: python scripts/realtime_listener.py
3. Open http://localhost:8050 in a browser to see the live plot.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import socket
import threading
import multiprocessing

import msgpack
import numpy as np
import tkinter as tk

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from flask import Flask

from src.io.pose_stamped import PoseStamped
from src.signal.filters import ButterLowPass_RT, ButterHighPass_RT
from src.algorithm.detector import GaitEventDetector
from src.algorithm.calibration import load_offsets

# ---------- Config ----------
UDP_IP = "127.0.0.1"
UDP_PORT = 8051
FS = 100
OFFSETS_PATH = os.path.join(os.path.dirname(__file__), "..", "offsets.json")

# Bandpass via LP(6 Hz) → HP(0.5 Hz) to match gross characteristics of offline filter
_lp_z = ButterLowPass_RT(cutoff=6, fs=FS)
_hp_z = ButterHighPass_RT(cutoff=0.5, order=2, fs=FS)
_lp_yr = ButterLowPass_RT(cutoff=6, fs=FS)
_hp_yr = ButterHighPass_RT(cutoff=0.5, order=2, fs=FS)


# ---------- UDP listener ----------

def udp_listener(cache, lock):
    offsets = load_offsets(OFFSETS_PATH)
    detector = GaitEventDetector(**offsets)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening on {UDP_IP}:{UDP_PORT} ...")

    unfiltered_z: list = []
    unfiltered_yr: list = []
    init_time = 0.0
    old_sync = 0
    offset_trans = None
    old_pose = None

    def safe_convert(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return x

    def process_packet(index, data):
        raw = msgpack.unpackb(data, raw=False)
        msg = [index]
        for item in raw["H"]:
            if item["Tag"] == "Pelvis":
                msg += item["P"] + item["O"] + [raw["T"]] + [item["Tag"]]
        return list(map(safe_convert, msg)), raw["Info"]["Sync"]

    def apply_rt_filter(buf, lp, hp):
        window = buf[-200:] if len(buf) >= 200 else buf
        return hp.apply(lp.apply(copy.copy(window))).tolist()

    while True:
        data, _ = sock.recvfrom(4096)
        try:
            with lock:
                idx = cache["index"]
                msg, current_sync = process_packet(idx, data)
                pose = PoseStamped(*msg)

                # Determine sync start time
                if init_time == 0.0 and old_sync == 0 and current_sync == 1:
                    init_time = pose.time
                old_sync = current_sync

                # Build coordinate offset on first frame after calibration reset
                if idx == 0:
                    rot = pose.rotation.T @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                    from transforms3d.affines import compose
                    offset_trans = compose([0.0, 0.0, 0.0], rot, [1.0, 1.0, 1.0])
                elif offset_trans is not None:
                    pose.normalize(init_time, offset_trans)

                # Accumulate and filter Z position
                unfiltered_z.append(pose.get_lin_z())
                filtered_z_series = apply_rt_filter(unfiltered_z, _lp_z, _hp_z)
                z_val = filtered_z_series[-1]

                # Accumulate and filter Y rotation
                unfiltered_yr.append(pose.get_y_rot())
                filtered_yr_series = apply_rt_filter(unfiltered_yr, _lp_yr, _hp_yr)
                yr_val = filtered_yr_series[-1]

                # Update rolling display buffer
                fz_buf = cache["filtered_z"]
                t_buf = cache["time"]
                fz_buf.append(z_val)
                t_buf.append(float(pose.time))
                if len(fz_buf) > 500:
                    fz_buf.pop(0)
                    t_buf.pop(0)
                cache["filtered_z"] = fz_buf
                cache["time"] = t_buf

                # Run detector
                if len(unfiltered_z) > 50:
                    new_events = detector.detect_step(z_val, yr_val)
                    for ev in new_events:
                        lst = cache[ev]
                        lst.append(float(pose.time))
                        cache[ev] = lst
                        if ev in ("LHS", "LTO"):
                            cache["L_state"] = 1 if ev == "LHS" else 0
                        else:
                            cache["R_state"] = 1 if ev == "RHS" else 0

                cache["index"] = idx + 1
                old_pose = pose

        except (ValueError, KeyError) as exc:
            print(f"Packet error: {exc}")


# ---------- Dash app ----------

def run_dash(cache, lock):
    server = Flask(__name__)
    app = dash.Dash(__name__, server=server)

    app.layout = html.Div([
        dcc.Interval(id="interval", interval=100, n_intervals=0),
        html.H3("Live Gait Event Detection"),
        html.Div(id="output", style={"fontSize": "20px", "marginBottom": "10px"}),
        dcc.Graph(id="live-plot"),
        html.Div(
            [
                html.Div("L", id="left-circle",
                         style={"width": "60px", "height": "60px", "borderRadius": "50%",
                                "backgroundColor": "green", "display": "inline-flex",
                                "alignItems": "center", "justifyContent": "center",
                                "fontSize": "20px", "color": "white"}),
                html.Div("R", id="right-circle",
                         style={"width": "60px", "height": "60px", "borderRadius": "50%",
                                "backgroundColor": "green", "display": "inline-flex",
                                "alignItems": "center", "justifyContent": "center",
                                "fontSize": "20px", "color": "white"}),
            ],
            style={"display": "flex", "justifyContent": "center", "gap": "20px", "marginTop": "20px"},
        ),
    ], style={"textAlign": "center"})

    @app.callback(
        [Output("output", "children"),
         Output("live-plot", "figure"),
         Output("left-circle", "style"),
         Output("right-circle", "style")],
        [Input("interval", "n_intervals")],
    )
    def update(n):
        with lock:
            fz = cache.get("filtered_z", [])
            t = cache.get("time", [])
            L = cache.get("L_state", 0)
            R = cache.get("R_state", 0)

        latest = f"{fz[-1]:.4f} m" if fz else "waiting..."

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=fz, mode="markers", name="Filtered Z"))
        fig.update_layout(title="Filtered Z Position (pelvis vertical)",
                          xaxis_title="Time (s)", yaxis_title="Z (m)",
                          template="plotly_dark")

        circle_base = {"width": "60px", "height": "60px", "borderRadius": "50%",
                       "display": "inline-flex", "alignItems": "center",
                       "justifyContent": "center", "fontSize": "20px", "color": "white"}
        L_style = {**circle_base, "backgroundColor": "red" if L else "green"}
        R_style = {**circle_base, "backgroundColor": "red" if R else "green"}

        return latest, fig, L_style, R_style

    app.run_server(debug=False, use_reloader=False, host="0.0.0.0", port=8050, threaded=True)


# ---------- Tkinter toolkit ----------

def run_toolkit(cache, lock, detector_ref):
    def calibrate():
        with lock:
            cache["index"] = 0
            cache["LHS"] = []; cache["RHS"] = []
            cache["LTO"] = []; cache["RTO"] = []
            cache["L_state"] = 0; cache["R_state"] = 0
        detector_ref[0].reset()
        print("Calibrated — detector reset.")

    def print_events():
        with lock:
            for ev in ("LHS", "RHS", "LTO", "RTO"):
                print(f"{ev}: {cache[ev]}")

    root = tk.Tk()
    root.title("Gait Event Detector Toolkit")
    root.geometry("300x160")
    tk.Button(root, text="Calibrate", command=calibrate).pack(pady=15)
    tk.Button(root, text="Print Gait Events", command=print_events).pack(pady=10)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


# ---------- Entry point ----------

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    cache = manager.dict()
    lock = manager.Lock()

    cache["LHS"] = []; cache["RHS"] = []
    cache["LTO"] = []; cache["RTO"] = []
    cache["L_state"] = 0; cache["R_state"] = 0
    cache["filtered_z"] = []; cache["time"] = []
    cache["index"] = 0

    offsets = load_offsets(OFFSETS_PATH)
    detector = GaitEventDetector(**offsets)
    detector_ref = [detector]  # mutable reference for toolkit reset callback

    listener_thread = threading.Thread(
        target=udp_listener, args=(cache, lock), daemon=True
    )
    listener_thread.start()

    toolkit_thread = threading.Thread(
        target=run_toolkit, args=(cache, lock, detector_ref), daemon=True
    )
    toolkit_thread.start()

    run_dash(cache, lock)
