!# Reproduction Walkthrough

Step-by-step guide to run the full offline pipeline and real-time system with real data.

---

## Step 0 — Environment

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Key packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `msgpack`, `transforms3d`, `dash`, `plotly`, `flask`.

---

## Step 1 — Prepare Your Data Files

You need two synchronized recordings from one trial:

### 1a. VIVE tracker data → CSV

**If you have raw `.msgpack` files:**

```python
from src.io.vive_loader import convert_msgpack_folder
convert_msgpack_folder("path/to/msgpack/folder")
# Writes *_expanded.csv alongside each .msgpack file
```

**Expected CSV column format** (produced by `convert_msgpack_folder` or exported directly from SteamVR):

| Column pattern | Meaning |
|---|---|
| `T` | Timestamp (seconds) |
| `Sync` | 1 = synced with mat, 0 = not |
| `P1 (Pelvis)`, `P2 (Pelvis)`, `P3 (Pelvis)` | Pelvis X, Y, Z position |
| `O1 (Pelvis)` … `O4 (Pelvis)` | Pelvis quaternion [w, x, y, z] |
| `P1 (Tracker_01)` … `P3 (Tracker_01)` | Corner tracker position |
| *(same pattern for Tracker_02, Tracker_03, Tracker_05)* | |

Save the final CSV to `data/vive_sample.csv`.

### 1b. Pressure mat data → CSV

**If you have raw `.txt` mat files:**

```python
from src.io.mat_loader import convert_mat_txt_folder
convert_mat_txt_folder("path/to/mat/folder")
# Writes _*.csv alongside each .txt file
```

**Expected CSV column format** after conversion (or if already exported):
- Row 0–14: header rows (skipped automatically by `load_mat_csv`)
- Remaining rows: one event per row
  - Column 1: side label (must contain "Left" or "Right")
  - Column 2: heel strike timestamp (relative, seconds)
  - Column 3: toe off timestamp (relative, seconds)

Save the final CSV to `data/mat_sample.csv`.

---

## Step 2 — Configure `main.py`

Open `main.py` and update the path and tracker constants at the top:

```python
VIVE_PATH    = "data/vive_sample.csv"   # ← your VIVE CSV
MAT_PATH     = "data/mat_sample.csv"    # ← your mat CSV
OFFSETS_PATH = "offsets.json"

PELVIS_TAG   = "Pelvis"                 # tag name in the VIVE CSV header
CORNER_TAGS  = ["Tracker_01", "Tracker_02", "Tracker_03", "Tracker_05"]  # mat-corner trackers

VIVE_FILTER_CUTOFF   = 6    # Hz — first-pass Butterworth on raw VIVE data
SIGNAL_FILTER_CUTOFF = 0.1  # Hz — second-pass smoothing for detection signals
FS = 100  # Hz — VIVE sampling rate
```

Adjust `CORNER_TAGS` to match the tag names in your CSV. The order matters:
- `CORNER_TAGS[0]` = corner used as the coordinate origin (top-right)
- `CORNER_TAGS[3]` = bottom-right corner (defines the X axis)
- `CORNER_TAGS[1]` = top-left corner (defines the Y axis)

---

## Step 3 — Run the Offline Pipeline

```bash
python main.py
```

This runs all steps in sequence:

1. Load VIVE and mat CSVs
2. Apply Butterworth filter to raw VIVE data
3. Compute lab coordinate frame from corner trackers
4. Transform pelvis pose into lab frame
5. Extract Z position and Y rotation signals, apply smoothing
6. Load ground-truth gait events from pressure mat
7. **Compute calibration offsets** → saves to `offsets.json`
8. **Detect gait events** using the calibrated detector
9. **Evaluate** (precision, recall, timing error per event type) → printed to terminal
10. **Visualize** (opens matplotlib windows):
    - 2D pelvis trajectory over the mat
    - Z/Y/X position over time with ground-truth event markers
    - Detected vs. ground-truth comparison plot

### What to check

- The calibration offsets in `offsets.json` should be small integers (typically 2–15 samples at 100 Hz). If they are 0 or very large, there is likely a timing alignment issue between the VIVE and mat data.
- Precision and recall should be > 0.8 for a well-calibrated trial. Lower values indicate poor signal quality or a mismatch in time alignment.
- In the position-with-events plot, ground-truth HS markers should visually align with Z troughs.

---

## Step 4 — Inspect Calibration Offsets

After `main.py` runs, `offsets.json` will contain your subject's calibration values:

```json
{
  "offset1": <samples from HS candidate to Z trough>,
  "offset2": <samples to search backward for a trough at TO>,
  "offset3": <Y-rotation peak search radius around each HS>,
  "offset4": <samples from each HS to its paired TO>
}
```

If you want to tune these manually, edit `offsets.json` directly and re-run from step 8 onwards (skip by commenting out `compute_offsets` and `save_offsets` in `main.py`).

---

## Step 5 — Real-Time System (Optional)

Requires the VIVE tracker streaming live pose data over UDP.

### Prerequisites

- `offsets.json` must already exist (produced by Step 3 on a calibration trial)
- Your VIVE system must be sending msgpack-encoded UDP packets to `127.0.0.1:8051`

### Start the listener

```bash
python scripts/realtime_listener.py
```

Then open `http://localhost:8050` in a browser.

### What you see

- **Live plot**: filtered Z position (pelvis vertical) scrolling in real time
- **L / R circles**: green = swing phase, red = stance phase
- **Tkinter window**: "Calibrate" resets the detector state; "Print Gait Events" dumps all detected event timestamps to the terminal

### If the detection looks wrong

1. Click "Calibrate" in the Tkinter window to reset state
2. Run a fresh calibration trial with `main.py` to regenerate `offsets.json`
3. Restart the listener

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| All offsets remain 0 after main.py | Mat events not aligning to VIVE time | Check `T[0]` vs mat timestamps; adjust `extract_mat_events` `estimated = T[0] + times` |
| Precision/recall = 0 | Wrong tracker tag names | Verify `PELVIS_TAG` and `CORNER_TAGS` match CSV headers exactly |
| Z signal looks flat | Lab frame transform wrong | Confirm corner tracker order; check that Z axis is vertical in your setup |
| RT detection fires constantly | Offsets are 0 | Run offline calibration first to populate offsets.json |
| msgpack import error | Package not installed | `pip install msgpack` |
| transforms3d import error | Package not installed | `pip install transforms3d` |
