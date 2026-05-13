# Gait Event Detection from a Single Pelvis-Mounted IMU/Tracker

A research project developing a real-time algorithm to detect four key gait events — **Left/Right Heel Strike (HS)** and **Left/Right Toe Off (TO)** — using only a single HTC VIVE tracker mounted on the pelvis. No foot-worn sensors required.

Collaborated with Guanjie Chen. **My contribution** covers the data analysis, pattern discovery, algorithm design, and offline validation pipeline. Guanjie's contribution covers signal filtering design and real-time system implementation.

---

## Motivation

Gait event detection is fundamental to clinical gait analysis and rehabilitation robotics. Traditional methods require pressure mats or foot-worn inertial sensors. This project investigates whether a **single pelvis tracker** — already present in many VR-based rehabilitation systems — contains sufficient information to recover all four gait events. If so, no additional hardware is needed.

---

## My Contribution: Algorithm Design

### Pattern Discovery

By analyzing synchronized VIVE tracker and pressure mat data across multiple walking trials (with and without a walker), I identified two biomechanical signals derivable from a pelvis-mounted tracker that together encode all four gait events:

1. **Vertical pelvis position (Z)**: The pelvis dips to a local minimum (trough) at every heel strike.
2. **Pelvis X angular velocity**: The sign of the pelvis rotation velocity at each trough discriminates left from right foot contact.

This observation allowed the design of a single-sensor detection algorithm without any foot instrumentation.

### Gait Event Detection Algorithm

The algorithm processes pelvis pose data frame-by-frame and classifies each frame as one of four gait events using two signals: vertical pelvis position (`z_pos`) and pelvis Y rotation (`y_rot`).

**Preprocessing** (per frame):
1. Apply 4th-order Butterworth filter to raw position and orientation data
2. Apply coordinate transformation using the ground tracker; convert quaternions to Euler angles
3. Smooth the data
4. Compute discrete gradients: `z_grad[n] = z_pos[n] - z_pos[n-1]`, `y_grad[n] = y_rot[n] - y_rot[n-1]`

**Heel Strike detection** (downward trend branch, `z_grad[n] < 0`):
- Look ahead by `offset1` samples: if `z_grad[n + offset1] > 0`, a Z trough is confirmed → candidate **HS** at index `n`
- Search forward and backward around `n` in the Y rotation signal: if a peak occurs within `|offset3|` samples → classify as **RHS**, otherwise **LHS**

**Toe Off detection** (upward trend branch, `z_grad[n] > 0`):
- Search backward within `offset2` samples: if a Z trough exists → candidate **TO** at index `n`
- Search forward within `offset4` samples for the paired HS:
  - If `z_pos[n - offset4] = RHS` → classify as **LTO**
  - If `z_pos[n - offset4] = LHS` → classify as **RTO**

### Calibration Offsets

The four offsets encode the biomechanical timing relationships discovered through data analysis against the pressure mat ground truth:

| Offset | Definition |
|--------|------------|
| Offset 1 | Sample difference between HS and the preceding Z trough |
| Offset 2 | Sample difference between TO and the preceding Z trough |
| Offset 3 | Sample difference between RHS and the nearest Y rotation peak |
| Offset 4 | Sample difference between HS and its paired TO |

These offsets are computed per subject during a calibration trial and used by the detection algorithm to make precise, temporally-accurate gait event predictions.

---

## Offline Analysis Pipeline (`src/analysis/`)

My pipeline for data exploration and algorithm validation:

| File | Description |
|------|-------------|
| `txt2csv.py` | Parses raw pressure mat `.txt` files into structured CSV |
| `msg2csv.py` | Decodes VIVE `.msgpack` binary recordings into CSV |
| `data_utils.py` | Signal processing utilities: Butterworth filtering, quaternion-to-rotation-matrix conversion, coordinate frame transformation, outlier removal, event-to-index mapping |
| `data_visual.py` | Full analysis pipeline: loads VIVE and mat data, applies coordinate transformation, visualizes pelvis trajectory in 2D/3D, overlays ground-truth gait events on position-over-time plots |
| `calibration.py` | Computes calibration offsets by detecting Z troughs and Y rotation peaks in sensor data and correlating them to ground-truth heel strike / toe off times from the pressure mat |

### Coordinate Transformation

The VIVE coordinate frame is arbitrary. I defined a lab coordinate frame using three mat-corner trackers and computed the rigid body transformation:

```
x̂_lab = (T_corner2 - T_corner1) / ‖...‖
ŷ_lab = orthogonal projection of third tracker onto x̂_lab plane
ẑ_lab = x̂_lab × ŷ_lab
T = [R | t]  where t = -R · origin
```

All pelvis positions and rotations are expressed in this consistent lab frame before analysis.

---

## Hardware Setup

| Component | Role |
|-----------|------|
| HTC VIVE Tracker (Tracker_04, pelvis) | Source of pelvis pose at ~100 Hz |
| HTC VIVE Trackers (Trackers 01, 02, 03, 05) | Mounted at mat corners to define the lab coordinate frame |
| Force-sensitive pressure mat (MAT system) | Ground truth for heel strike and toe off timestamps |

---

## Real-Time System (Guanjie's contribution)

The algorithm runs in real time via a three-thread architecture:
- **UDP Listener**: receives VIVE tracker pose data (msgpack over UDP), applies the gait event algorithm, updates shared state
- **Dash Web App**: live plot of filtered Z signal + left/right foot state indicators (red = stance, green = swing), refreshed at 10 Hz
- **Tkinter Toolkit**: GUI controls for calibration reset and event export

---

## Tech Stack

- **Python** — NumPy, SciPy, pandas, transforms3d
- **Signal Processing** — Butterworth low-pass/high-pass filters, gradient-based feature detection
- **3D Geometry** — Quaternion-to-rotation-matrix conversion, rigid body transformation, angular velocity from rotation matrix derivative
- **Data Visualization** — Matplotlib (offline), Plotly/Dash (real-time)
- **Hardware Interface** — HTC VIVE SteamVR, UDP socket, msgpack serialization
- **Concurrency** — Python threading + multiprocessing with shared cache and mutex locks

---

## Repository Structure

```
├── main.py                         # Offline analysis pipeline (load → calibrate → detect → evaluate → plot)
├── offsets.json                    # Calibrated offsets (written by main.py, read by realtime_listener.py)
├── scripts/
│   └── realtime_listener.py        # Real-time pipeline: UDP → detect_step() → Dash + Tkinter
├── src/
│   ├── io/
│   │   ├── vive_loader.py          # VIVE CSV loader; msgpack → CSV conversion
│   │   ├── mat_loader.py           # Pressure mat CSV loader; .txt → CSV conversion
│   │   └── pose_stamped.py         # PoseStamped: coordinate transform, Z position, Y rotation
│   ├── signal/
│   │   ├── filters.py              # Offline (filtfilt) and real-time (lfilter) Butterworth filters
│   │   └── transforms.py           # Quaternion → rotation matrix, lab frame, coordinate transform, event alignment
│   ├── algorithm/
│   │   ├── detector.py             # GaitEventDetector: detect() (offline) + detect_step() (real-time)
│   │   └── calibration.py          # compute_offsets(), save_offsets(), load_offsets()
│   ├── evaluation/
│   │   └── metrics.py              # precision_recall(), timing_errors(), print_report()
│   └── visualization/
│       └── plots.py                # 2D/3D trajectory, position-with-events, detected vs ground truth
├── algorithms/
│   ├── Real-Time Algorithm Pipeline.drawio   # Gait event detection flowchart
│   └── Calibration Algorithm.drawio          # Calibration offset computation flowchart
└── data/                           # Sample synchronized VIVE + pressure mat recordings
```
