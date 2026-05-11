"""
Gait Event Detection — Analysis Pipeline

Full pipeline:
  1. Load synchronized VIVE tracker and pressure mat data
  2. Preprocess: filter, coordinate transform, extract signals
  3. Compute calibration offsets from ground-truth mat events
  4. Detect gait events using the calibrated algorithm
  5. Visualize results

Usage
-----
Update VIVE_PATH and MAT_PATH to point to your data files, then run:
    python main.py
"""

import numpy as np

from src.data_loader import load_vive_csv, load_mat_csv
from src.preprocessing import (
    apply_butterworth_to_df,
    rotation_matrices_from_df,
    rotation_to_euler,
    apply_transform,
    compute_lab_frame,
    mean_position,
    extract_mat_events,
)
from src.calibration import compute_offsets
from src.gait_event_detector import GaitEventDetector
from src.visualization import (
    plot_2D_trajectory,
    plot_position_with_events,
    plot_detected_vs_ground_truth,
)

# ---------- Paths ----------
VIVE_PATH = "data/vive_sample.csv"
MAT_PATH  = "data/mat_sample.csv"

# ---------- Tracker column names ----------
PELVIS_TAG = "Pelvis"
CORNER_TAGS = ["Tracker_01", "Tracker_02", "Tracker_03", "Tracker_05"]

VIVE_FILTER_CUTOFF = 6   # Hz — applied to raw VIVE data
SIGNAL_FILTER_CUTOFF = 0.1  # Hz — applied to pelvis position/orientation for detection
FS = 100  # Hz


def main():
    # 1. Load data
    vive_df = load_vive_csv(VIVE_PATH)
    mat_df  = load_mat_csv(MAT_PATH)

    # 2. Filter raw VIVE data
    numeric_cols = vive_df.select_dtypes(include="number").columns.tolist()
    vive_df[numeric_cols] = apply_butterworth_to_df(
        vive_df[numeric_cols], cutoff=VIVE_FILTER_CUTOFF, fs=FS
    )

    # 3. Extract time array
    T = vive_df["T"].to_numpy().astype(float)

    # 4. Extract pelvis position and orientation
    pelvis_pos  = vive_df[[f"P{i} ({PELVIS_TAG})" for i in range(1, 4)]].to_numpy()
    pelvis_quat_cols = [f"O{i} ({PELVIS_TAG})" for i in range(1, 5)]
    pelvis_R = rotation_matrices_from_df(vive_df, pelvis_quat_cols)

    # 5. Extract corner tracker positions and compute lab frame
    corner_means = []
    for tag in CORNER_TAGS:
        pos = vive_df[[f"P{i} ({tag})" for i in range(1, 4)]].to_numpy()
        corner_means.append(mean_position(pos))

    _, T_lab = compute_lab_frame(corner_means[0], corner_means[3], corner_means[1])

    # 6. Transform pelvis into lab frame
    pelvis_R_lab, pelvis_pos_lab = apply_transform(pelvis_R, pelvis_pos, T_lab)

    # 7. Extract Z position and Y rotation signals
    pelvis_x = pelvis_pos_lab[:, 0]
    pelvis_y = pelvis_pos_lab[:, 1]
    pelvis_z = pelvis_pos_lab[:, 2]

    euler_angles = rotation_to_euler(pelvis_R_lab)  # (N, 3): [roll, pitch, yaw]
    pelvis_y_rot = euler_angles[:, 1]               # pitch = Y rotation

    # 8. Additional low-pass smoothing for the detection signals
    from src.preprocessing import butterworth_filter
    z_smooth   = butterworth_filter(pelvis_z,     cutoff=SIGNAL_FILTER_CUTOFF, fs=FS)
    yrot_smooth = butterworth_filter(pelvis_y_rot, cutoff=SIGNAL_FILTER_CUTOFF, fs=FS)

    # 9. Load ground-truth gait events from pressure mat
    HS_left_idx, HS_right_idx, TO_left_idx, TO_right_idx = extract_mat_events(mat_df, T)

    # 10. Compute calibration offsets
    offsets = compute_offsets(
        z_smooth, yrot_smooth,
        HS_left_idx, HS_right_idx,
        TO_left_idx, TO_right_idx,
    )
    print("Calibration offsets:", offsets)

    # 11. Detect gait events
    detector = GaitEventDetector(**offsets)
    detected = detector.detect(z_smooth, yrot_smooth)
    print("Detected events (sample indices):")
    for event, indices in detected.items():
        print(f"  {event}: {len(indices)} events")

    # 12. Visualize
    ground_truth = {
        "LHS": list(HS_left_idx),
        "RHS": list(HS_right_idx),
        "LTO": list(TO_left_idx),
        "RTO": list(TO_right_idx),
    }

    corner_pos_list = [
        vive_df[[f"P{i} ({tag})" for i in range(1, 4)]].to_numpy()
        for tag in CORNER_TAGS
    ]
    plot_2D_trajectory(pelvis_pos_lab, corner_pos_list, tracker_labels=CORNER_TAGS)

    plot_position_with_events(
        T, pelvis_x, pelvis_y, pelvis_z,
        T[HS_left_idx], T[HS_right_idx], T[TO_left_idx], T[TO_right_idx],
        title_suffix="with Ground Truth Events",
    )

    plot_detected_vs_ground_truth(T, z_smooth, detected, ground_truth)


if __name__ == "__main__":
    main()
