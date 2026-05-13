import numpy as np
import matplotlib.pyplot as plt


def plot_2D_trajectory(pelvis_pos, tracker_positions, tracker_labels=None):
    """
    Top-down (X–Y) plot of the pelvis trajectory with mat corner tracker positions.

    Parameters
    ----------
    pelvis_pos       : (N, 3) array
    tracker_positions: list of (M, 3) arrays, one per ground tracker
    tracker_labels   : optional list of label strings
    """
    plt.figure(figsize=(8, 10))
    plt.plot(pelvis_pos[:, 1], pelvis_pos[:, 0], alpha=0.6, marker="o", markersize=2, label="Pelvis")

    for i, pos in enumerate(tracker_positions):
        mean_pos = np.mean(pos, axis=0)
        label = tracker_labels[i] if tracker_labels else f"Tracker {i + 1}"
        plt.scatter(mean_pos[1], mean_pos[0], s=100, label=label)

    plt.title("Pelvis Trajectory (Top-Down View)")
    plt.xlabel("Y Position (m)")
    plt.ylabel("X Position (m)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_3D_position(x, y, z, title="Pelvis 3D Trajectory"):
    """3D trajectory plot of pelvis position."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_3D_orientation(x_rot, y_rot, z_rot, title="Pelvis Orientation"):
    """3D trajectory of Euler angles."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_rot, y_rot, z_rot)
    ax.set_xlabel("Roll (rad)")
    ax.set_ylabel("Pitch (rad)")
    ax.set_zlabel("Yaw (rad)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_position_with_events(
    T, x, y, z,
    HS_left_times, HS_right_times,
    TO_left_times, TO_right_times,
    title_suffix="",
):
    """
    Three-panel time-series plot (X, Y, Z) with gait events as vertical lines.

    Parameters
    ----------
    T               : (N,) time array in seconds
    x, y, z         : (N,) position arrays
    *_times         : arrays of event timestamps in seconds
    title_suffix    : appended to each subplot title
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    labels = ["X", "Y", "Z"]
    signals = [x, y, z]

    event_styles = [
        (HS_left_times,  "grey",    "--", "Left Heel Strike"),
        (HS_right_times, "purple",  "--", "Right Heel Strike"),
        (TO_left_times,  "orange",  "--", "Left Toe Off"),
        (TO_right_times, "magenta", "--", "Right Toe Off"),
    ]

    for ax, sig, label in zip(axs, signals, labels):
        ax.plot(T, sig, color="black", linewidth=0.8)
        ax.set_ylabel(f"{label} (m)")
        ax.set_title(f"Pelvis {label} Position {title_suffix}")
        for times, color, style, name in event_styles:
            for t in times:
                ax.axvline(x=t, color=color, linestyle=style, linewidth=0.7, label=name)

    axs[-1].set_xlabel("Time (s)")

    handles, seen = [], set()
    for ax in axs:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h)
                seen.add(l)
    fig.legend(handles, list(seen), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def plot_detected_vs_ground_truth(
    T, z_pos,
    detected_events, ground_truth_events,
    title="Detected vs Ground Truth Gait Events",
):
    """
    Overlay detected gait events (solid lines) against ground truth (dashed)
    on the Z-position signal.

    Parameters
    ----------
    T                   : (N,) time array
    z_pos               : (N,) vertical pelvis position
    detected_events     : dict with keys LHS, RHS, LTO, RTO — lists of sample indices
    ground_truth_events : dict with same keys — lists of sample indices
    """
    colors = {"LHS": "grey", "RHS": "purple", "LTO": "orange", "RTO": "magenta"}

    plt.figure(figsize=(14, 4))
    plt.plot(T, z_pos, color="black", linewidth=0.8, label="Z position")

    for event, color in colors.items():
        for idx in detected_events.get(event, []):
            plt.axvline(T[idx], color=color, linewidth=1.2, label=f"{event} (detected)")
        for idx in ground_truth_events.get(event, []):
            plt.axvline(T[idx], color=color, linewidth=1.2, linestyle="--", label=f"{event} (ground truth)")

    handles, seen = [], set()
    for h, l in zip(*plt.gca().get_legend_handles_labels()):
        if l not in seen:
            handles.append(h)
            seen.add(l)
    plt.legend(handles, list(seen), loc="upper right", fontsize=8)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Z Position (m)")
    plt.tight_layout()
    plt.show()
