import numpy
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

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


def rotate_data(mx, my, mz, quaternions, ned=True):
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
    return normalize_quaternion(gradiant)


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


def slerp_quaternion(q1, q2, t):
    if np.isscalar(t):
        t = np.full(q1.shape[0], t)
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    dot = np.sum(q1 * q2, axis=1)
    neg_mask = dot < 0
    q2_adj = q2.copy()
    q2_adj[neg_mask] = -q2_adj[neg_mask]
    dot[neg_mask] = -dot[neg_mask]
    result = np.zeros_like(q1)
    close_mask = dot > 0.9995
    close_indices = np.where(close_mask)[0]
    if len(close_indices) > 0:
        t_close = t[close_indices].reshape(-1, 1)
        result[close_indices] = q1[close_indices] + t_close * (
                q2_adj[close_indices] - q1[close_indices]
        )
    not_close_indices = np.where(~close_mask)[0]
    if len(not_close_indices) > 0:
        dot_not_close = dot[not_close_indices]
        theta_0 = np.arccos(np.clip(dot_not_close, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        small_sin_mask = np.abs(sin_theta_0) < 1e-10
        small_indices = not_close_indices[small_sin_mask]
        if len(small_indices) > 0:
            t_small = t[small_indices].reshape(-1, 1)
            result[small_indices] = q1[small_indices] + t_small * (
                    q2_adj[small_indices] - q1[small_indices]
            )
        regular_mask = ~small_sin_mask
        regular_indices = not_close_indices[regular_mask]
        if len(regular_indices) > 0:
            t_regular = t[regular_indices]
            theta = theta_0[regular_mask] * t_regular
            sin_theta = np.sin(theta)
            sin_theta = sin_theta.reshape(-1, 1)
            theta_reshaped = theta.reshape(-1, 1)
            dot_regular = dot_not_close[regular_mask].reshape(-1, 1)
            sin_theta_0_regular = sin_theta_0[regular_mask].reshape(-1, 1)
            s0 = np.cos(theta_reshaped) - dot_regular * sin_theta / sin_theta_0_regular
            s1 = sin_theta / sin_theta_0_regular
            result[regular_indices] = (
                    s0 * q1[regular_indices] + s1 * q2_adj[regular_indices]
            )
    return normalize_quaternion(result)


def interp_quaternions(ts_to, ts_from, quaternions):
    res = []
    count = 0  # Counter to track how many ts_to values we’ve interpolated

    for i in range(len(ts_from) - 1):
        ts_n = ts_from[i]
        ts_np = ts_from[i + 1]
        mask = (ts_to >= ts_n) & (ts_to < ts_np)
        n = np.count_nonzero(mask)

        if n == 0:
            continue

        q1 = quaternions[i]
        q2 = quaternions[i + 1]

        t_values = np.linspace(0, 1, n, endpoint=False)
        qs = slerp_quaternion(
            np.repeat([q1], n, axis=0),
            np.repeat([q2], n, axis=0),
            t_values,
        )
        res.extend(qs)
        count += n

    # Add final quaternion to match the length of ts_to
    remaining = len(ts_to) - count
    if remaining > 0:
        q_last = normalize_quaternion(np.array([quaternions[-1]]))[0]
        res.extend([q_last] * remaining)

    return np.array(res)


def quaternion_to_euler(quaternions):  # For NED
    qw, qx, qy, qz = quaternions.T
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    return (np.degrees(yaw)) % 360 - 180, np.degrees(pitch), np.degrees(roll)


def euler_to_quaternion(yaw, pitch, roll):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def gnss_to_yaw(x_gnss, y_gnss, ts_gnss, duration_gnss_filter=2):
    res = numpy.arctan2(
        numpy.diff(x_gnss),
        numpy.diff(y_gnss)
    )
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
    x = scipy.interpolate.Akima1DInterpolator(ts_gnss, gnss_x)(ts)
    y = scipy.interpolate.Akima1DInterpolator(ts_gnss, gnss_y)(ts)
    z = scipy.interpolate.Akima1DInterpolator(ts_gnss, gnss_z)(ts)
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
            z + offsets_z
        ])

    p1x, p1y, p1z = numpy.array(sensors_positions[0])[0]
    p2x, p2y, p2z = numpy.array(sensors_positions[1])[0]
    p3x, p3y, p3z = numpy.array(sensors_positions[2])[0]
    p4x, p4y, p4z = numpy.array(sensors_positions[3])[0]

    coord_lambert_capteurs = numpy.array([
        p1x, p1y, p1z,
        p2x, p2y, p2z,
        p3x, p3y, p3z,
        p4x, p4y, p4z]).T
    coord_lambert_capteurs_echan = coord_lambert_capteurs[::200]

    for nn in range(len(coord_lambert_capteurs_echan)):
        coord_one_group = np.c_[coord_lambert_capteurs_echan[nn][0:13:3], coord_lambert_capteurs_echan[nn][1:13:3]]
        plt.plot(coord_one_group[:, 0], coord_one_group[:, 1], 'k-o')

    plt.plot(x, y)
    plt.plot(p1x, p1y)
    plt.plot(p2x, p2y)
    plt.plot(p3x, p3y)
    plt.plot(p4x, p4y)
    plt.axis("equal")
    plt.show()

    return ts_filtered, sensors_positions


def madgwick_step(quaternion, dt, coef, mag, gyr, acc, gain_acc, gain_mag, step, earth_vector):
    step_acc = quaternion_gradiant(quaternion, acc, np.array([0, 0, 1]))
    step_mag = quaternion_gradiant(quaternion, mag, earth_vector)
    step_gyr = product_quaternion(quaternion, step * gyr)
    step_total = 0.5 * step_gyr - (
            gain_acc * step_acc +
            gain_mag * step_mag
    ) * coef
    return normalize_quaternion(quaternion + dt * step_total)
