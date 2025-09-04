import numpy
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation, Slerp

import utils


def rotate_mag_old(mx, my, mz, quaternions):  # Rotate quaternions in ENU convention
    qw, qx, qy, qz = quaternions.T
    mx_rot = np.zeros_like(mx)
    my_rot = np.zeros_like(my)
    mz_rot = np.zeros_like(mz)
    for i in range(len(mx)):
        mx_rot[i] = (
            (1 - 2 * qy[i] * qy[i] - 2 * qz[i] * qz[i]) * mx[i] +
            (2 * qx[i] * qy[i] - 2 * qw[i] * qz[i]) * my[i] +
            (2 * qx[i] * qz[i] + 2 * qw[i] * qy[i]) * mz[i])

        my_rot[i] = (
            (2 * qx[i] * qy[i] + 2 * qw[i] * qz[i]) * mx[i] +
            (1 - 2 * qx[i] * qx[i] - 2 * qz[i] * qz[i]) * my[i] +
            (2 * qy[i] * qz[i] - 2 * qw[i] * qx[i]) * mz[i])

        mz_rot[i] = (
            (2 * qx[i] * qz[i] - 2 * qw[i] * qy[i]) * mx[i] +
            (2 * qy[i] * qz[i] + 2 * qw[i] * qx[i]) * my[i] +
            (1 - 2 * qx[i] * qx[i] - 2 * qy[i] * qy[i]) * mz[i])

    return mx_rot, my_rot, mz_rot


def avg_quaternion_from_mag(mx_ref, my_ref, mz_ref, mx, my, mz):
    ref = np.vstack([mx_ref, my_ref, mz_ref]).T
    meas = np.vstack([mx, my, mz]).T

    # Normalize
    ref /= np.linalg.norm(ref, axis=1)[:, None]
    meas /= np.linalg.norm(meas, axis=1)[:, None]

    # Correlation matrix
    M = meas.T @ ref

    # SVD
    U, _, Vt = np.linalg.svd(M)
    R_opt = U @ Vt

    # Ensure right-handed rotation
    if np.linalg.det(R_opt) < 0:
        U[:, -1] *= -1
        R_opt = U @ Vt

    # Convert to quaternion (x, y, z, w format in scipy)
    quat = Rotation.from_matrix(R_opt).as_quat()
    return numpy.array([quat[3], quat[0], quat[1], quat[2]])


def rotate_data(mx, my, mz, quaternions, ned=True):
    if len(numpy.shape(quaternions)) == 1:
        quaternions = np.tile(quaternions, (len(mx), 1))
    mags_rot = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).apply(np.stack((mx, my, mz), axis=-1))
    if ned:
        return mags_rot[:, 0], mags_rot[:, 1], mags_rot[:, 2]  # NED
    else:
        return mags_rot[:, 1], mags_rot[:, 0], -mags_rot[:, 2]  # ENU


def product_quaternion(q1, gyro):
    return np.array([
        -q1[1] * gyro[0] - q1[2] * gyro[1] - q1[3] * gyro[2],
        q1[0] * gyro[0] + q1[2] * gyro[2] - q1[3] * gyro[1],
        q1[0] * gyro[1] - q1[1] * gyro[2] + q1[3] * gyro[0],
        q1[0] * gyro[2] + q1[1] * gyro[1] - q1[2] * gyro[0],
    ])


def jacobian(q, g):
    gq0 = q[1] * g[0] + q[2] * g[1] + q[3] * g[2]
    gq1 = q[0] * g[0] + q[3] * g[1] - q[2] * g[2]
    gq2 = -q[3] * g[0] + q[0] * g[1] + q[1] * g[2]
    gq3 = q[2] * g[0] - q[1] * g[1] + q[0] * g[2]
    return 2 * np.array([
        [gq1, gq2, gq3],
        [gq0, gq3, -gq2],
        [-gq3, gq0, gq1],
        [gq2, -gq1, gq0]])


def quaternion_error(q, a, g):
    q01 = q[0] * q[1]
    q02 = q[0] * q[2]
    q03 = q[0] * q[3]
    q12 = q[1] * q[2]
    q13 = q[1] * q[3]
    q23 = q[2] * q[3]
    q11 = q[1] ** 2
    q22 = q[2] ** 2
    q33 = q[3] ** 2
    return np.array([
        g[0] - 2 * (g[0] * (q22 + q33) - g[1] * (q03 + q12) - g[2] * (q13 - q02)) - a[0],
        g[1] + 2 * (g[0] * (q12 - q03) - g[1] * (q11 + q33) + g[2] * (q01 + q23)) - a[1],
        g[2] + 2 * (g[0] * (q02 + q13) + g[1] * (q23 - q01) - g[2] * (q11 + q22)) - a[2]])


def quaternion_gradiant(quaternion, measure, reference):
    jacobian_matrix = jacobian(quaternion, reference)
    error = quaternion_error(quaternion, measure, reference)
    gradiant = np.dot(jacobian_matrix, error)
    return normalize_quaternion(gradiant)  # Très important de normaliser


def normalize_quaternion(q):
    q = np.asarray(q)

    if q.ndim == 1:  # Cas unique: shape (4,)
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm

    elif q.ndim == 2 and q.shape[1] == 4:  # Cas batch: shape (N, 4)
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        result = np.zeros_like(q)
        result = np.divide(q, norm, out=result, where=norm >= 1e-10)
        mask = (norm < 1e-10).flatten()
        if np.any(mask):
            result[mask] = np.array([1.0, 0.0, 0.0, 0.0])
        return result

    else:
        raise ValueError("Input must be a quaternion (4,) or array of shape (N, 4)")


def convert_wxyz_to_xyzw(q):
    """Convertit les quaternions du format [w, x, y, z] vers [x, y, z, w]."""
    return np.concatenate([q[:, 1:], q[:, [0]]], axis=1)


def convert_xyzw_to_wxyz(q):
    """Convertit les quaternions du format [x, y, z, w] vers [w, x, y, z]."""
    return np.concatenate([q[:, [3]], q[:, :3]], axis=1)


def slerp_quaternion(q1, q2, t):
    if np.isscalar(t):
        t = np.full(q1.shape[0], t)
    result = np.zeros_like(q1)
    q1_xyzw = convert_wxyz_to_xyzw(q1)
    q2_xyzw = convert_wxyz_to_xyzw(q2)
    for i in range(q1.shape[0]):
        key_times = [0, 1]
        key_rots = Rotation.from_quat([q1_xyzw[i], q2_xyzw[i]])
        slerp = Slerp(key_times, key_rots)
        interp_rot = slerp([t[i]])
        result_xyzw = interp_rot.as_quat()[0]
        result[i] = convert_xyzw_to_wxyz(np.array([result_xyzw]))[0]
    return result


def quaternion_to_euler(quaternions):  # For NED
    qw, qx, qy, qz = quaternions.T
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    return (np.degrees(yaw)) % 360 - 180, np.degrees(pitch), np.degrees(roll)


def gnss_to_yaw(x_gnss, y_gnss, ts_gnss, duration_gnss_filter=2):
    res = numpy.arctan2(
        numpy.diff(x_gnss),
        numpy.diff(y_gnss))
    res = utils.low_pass(res, ts=ts_gnss, t=duration_gnss_filter)
    return np.degrees(res)


def rot_mat_to_quaternions(rot_mat):
    R = np.atleast_3d(rot_mat)  # Assure que R est de forme (N, 3, 3)
    N = R.shape[0]
    quaternions = np.empty((N, 4))
    for i in range(N):
        m = R[i]
        trace = np.trace(m)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2  # S=4*qw
            qw = 0.25 * S
            qx = (m[2, 1] - m[1, 2]) / S
            qy = (m[0, 2] - m[2, 0]) / S
            qz = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2  # S=4*qx
            qw = (m[2, 1] - m[1, 2]) / S
            qx = 0.25 * S
            qy = (m[0, 1] + m[1, 0]) / S
            qz = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2  # S=4*qy
            qw = (m[0, 2] - m[2, 0]) / S
            qx = (m[0, 1] + m[1, 0]) / S
            qy = 0.25 * S
            qz = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2  # S=4*qz
            qw = (m[1, 0] - m[0, 1]) / S
            qx = (m[0, 2] + m[2, 0]) / S
            qy = (m[1, 2] + m[2, 1]) / S
            qz = 0.25 * S
        quaternions[i] = [qw, qx, qy, qz]
    return quaternions


def mag_position(ts_gnss, gnss_x, gnss_y, gnss_z, ts, quaternions, positions=np.array([-0.80353, -0.27744, 0.27744, 0.80353])):
    mask_gnss = (ts_gnss <= numpy.max(ts)) & (ts_gnss >= numpy.min(ts))
    x = scipy.interpolate.Akima1DInterpolator(ts_gnss[mask_gnss], gnss_x[mask_gnss])(ts)
    y = scipy.interpolate.Akima1DInterpolator(ts_gnss[mask_gnss], gnss_y[mask_gnss])(ts)
    z = scipy.interpolate.Akima1DInterpolator(ts_gnss[mask_gnss], gnss_z[mask_gnss])(ts)
    mask = ~np.isnan(numpy.array([x, y, z])).any(axis=0)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    quaternions = quaternions[mask]
    ts_filtered = ts[mask]

    sensors_positions = []

    for j, offset in enumerate(positions):
        sensors_positions.append([])
        offsets_x, offsets_y, offsets_z = rotate_data(0, offset, 0, quaternions, ned=False)
        sensors_positions[j].append([
            x + offsets_x,
            y + offsets_y,
            z + offsets_z])

    return ts_filtered, sensors_positions, mask


def madgwick_step(quaternion, dt, coef, mag, gyr, acc, gain_acc, gain_mag, step, earth_vector, beta, normalize_madgwick_step):
    step_acc = quaternion_gradiant(quaternion, acc, np.array([0, 0, 1]))  # Normalisé
    step_mag = quaternion_gradiant(quaternion, mag, earth_vector)  # Normalisé
    step_gyr = product_quaternion(quaternion, step * gyr)  # Non-Normalisé
    step_total = beta * (0.5 * step_gyr - (
            gain_acc * step_acc +
            gain_mag * step_mag
    ) * coef)  # Non-Normalisé
    res = quaternion + dt * step_total
    if normalize_madgwick_step:
        res = normalize_quaternion(res)
    return res


def low_pass_quaternions(quaternions, ts, duration):
    if duration == 0:
        return quaternions
    smoothed = np.zeros_like(quaternions)
    smoothed[0] = quaternions[0]
    smoothing_factor = 1 - 1 / (duration / numpy.median(numpy.diff(ts)) + 1)
    for i in range(1, len(quaternions)):
        smoothed[i] = slerp_quaternion(numpy.array([smoothed[i - 1]]), numpy.array([quaternions[i]]), smoothing_factor)
    return smoothed
