import numpy as np
from scipy.spatial.transform import Rotation
from transforms3d.quaternions import quat2mat
from transforms3d.affines import compose


def form_trans_matrix(position, orientation):
    rotation_matrix = quat2mat(orientation.flatten())
    translation = position.flatten()
    return compose(translation, rotation_matrix, [1.0, 1.0, 1.0])


class PoseStamped:
    def __init__(self, index, px, py, pz, ow, ox, oy, oz, t, tracker_id):
        self.position = np.array([[px], [py], [pz]])
        self.orientation = np.array([[ow], [ox], [oy], [oz]])  # wxyz
        self.rotation = quat2mat(self.orientation.flatten())
        self.time = t
        self.T = form_trans_matrix(self.position, self.orientation)
        self.index = index
        self.tracker_id = tracker_id

    def normalize(self, init_time, offset):
        self.time = self.time - init_time
        self.T = self.T @ offset

    def get_lin_z(self):
        return float(self.position[2, 0])

    def get_y_rot(self):
        """Pelvis pitch (Y Euler angle) in the body frame."""
        return Rotation.from_matrix(self.rotation).as_euler("xyz")[1]

    def get_ang_x(self, T_t2, dt):
        """X angular velocity from successive rotation matrices (body frame)."""
        R1 = self.T[:3, :3]
        R2 = T_t2[:3, :3]
        R_dot = (R2 - R1) / dt
        omega = R1.T @ R_dot
        angular_vel = np.array([
            (omega[2, 1] - omega[1, 2]) / 2,
            (omega[0, 2] - omega[2, 0]) / 2,
            (omega[1, 0] - omega[0, 1]) / 2,
        ])
        return float(angular_vel[0])
