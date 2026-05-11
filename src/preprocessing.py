import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation
from scipy.stats import zscore


# ---------- Filtering ----------

def butterworth_filter(data, cutoff, fs, order=4):
    """Zero-phase Butterworth low-pass filter."""
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype="low", analog=False)
    return filtfilt(b, a, data)


def apply_butterworth_to_df(df, cutoff, fs, order=4):
    """Apply Butterworth filter to every column in a DataFrame."""
    result = df.copy()
    for col in result.columns:
        result[col] = butterworth_filter(result[col].values, cutoff, fs, order)
    return result


# ---------- Quaternion / rotation ----------

def quat_to_rotation_matrix(q):
    """Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    # scipy expects [x, y, z, w]
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    return Rotation.from_quat(q_xyzw).as_matrix()


def rotation_matrices_from_df(df, quat_columns):
    """
    Build an (N, 3, 3) array of rotation matrices from four quaternion columns.
    quat_columns: [w_col, x_col, y_col, z_col]
    """
    quats = df[quat_columns].to_numpy()
    R = np.zeros((len(quats), 3, 3))
    for i, q in enumerate(quats):
        R[i] = quat_to_rotation_matrix(q)
    return R


def rotation_to_euler(R_array):
    """
    Convert an (N, 3, 3) rotation matrix array to Euler angles (radians).
    Returns (N, 3) array [roll, pitch, yaw].
    """
    angles = np.zeros((len(R_array), 3))
    for i, R in enumerate(R_array):
        angles[i] = Rotation.from_matrix(R).as_euler("xyz")
    return angles


# ---------- Coordinate frame ----------

def compute_lab_frame(corner_TR, corner_TL, corner_BR):
    """
    Compute a transformation matrix that maps VIVE world coordinates into the
    lab/mat coordinate frame defined by three mat-corner tracker positions.

    Parameters
    ----------
    corner_TR : (3,) array — top-right corner tracker mean position
    corner_TL : (3,) array — top-left corner tracker mean position
    corner_BR : (3,) array — bottom-right corner tracker mean position

    Returns
    -------
    origin : (3,) array
    T      : (4, 4) homogeneous transformation matrix
    """
    origin = corner_TR
    x_axis = (corner_BR - corner_TR) / np.linalg.norm(corner_BR - corner_TR)
    y_proj = corner_TL - corner_TR - np.dot(corner_TL - corner_TR, x_axis) * x_axis
    y_axis = y_proj / np.linalg.norm(y_proj)
    z_axis = np.cross(x_axis, y_axis)

    R = np.vstack([x_axis, y_axis, z_axis])
    t = -R @ origin

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return origin, T


def apply_transform(R_array, p_array, T):
    """
    Apply a (4, 4) homogeneous transformation to arrays of rotation matrices and positions.

    Parameters
    ----------
    R_array : (N, 3, 3)
    p_array : (N, 3)
    T       : (4, 4)

    Returns
    -------
    R_transformed : (N, 3, 3)
    p_transformed : (N, 3)
    """
    R_transformed = np.zeros_like(R_array)
    p_transformed = np.zeros_like(p_array)

    for i in range(len(R_array)):
        H = np.eye(4)
        H[:3, :3] = R_array[i]
        H_out = T @ H
        R_transformed[i] = H_out[:3, :3]

        p_h = np.append(p_array[i], 1.0)
        p_transformed[i] = (T @ p_h)[:3]

    return R_transformed, p_transformed


# ---------- Outlier removal / statistics ----------

def remove_outliers(data, threshold=3):
    """Remove rows where any column exceeds `threshold` standard deviations."""
    abs_z = np.abs(zscore(data))
    mask = (abs_z < threshold).all(axis=1)
    return data[mask]


def mean_position(data, n_samples=1000):
    """Estimate mean tracker position from a random subsample after outlier removal."""
    indices = np.random.choice(len(data), size=min(n_samples, len(data)), replace=False)
    sample = data[indices]
    cleaned = remove_outliers(sample)
    return np.mean(cleaned, axis=0)


# ---------- Event alignment ----------

def map_events_to_indices(event_times, T, tolerance=0.007):
    """
    Map an array of event timestamps to their nearest index in time array T.
    Events without a match within `tolerance` seconds are assigned index 0.
    """
    indices = {}
    for i, t_event in enumerate(event_times):
        for j, t in enumerate(T):
            if abs(t - t_event) <= tolerance:
                indices[i] = j
                break
    return np.array([indices.get(i, 0) for i in range(len(event_times))], dtype=int)


def extract_mat_events(mat_df, T):
    """
    Extract heel strike and toe off indices from the mat DataFrame,
    aligned to the VIVE time array T.

    Returns
    -------
    HS_left_idx, HS_right_idx, TO_left_idx, TO_right_idx : int arrays
    """
    def _get(side, col_offset):
        rows = mat_df[mat_df["Side"] == side]
        times = rows.iloc[:, col_offset].to_numpy().astype(float)
        estimated = T[0] + times
        return map_events_to_indices(estimated, T)

    return _get("left", 2), _get("right", 2), _get("left", 3), _get("right", 3)
