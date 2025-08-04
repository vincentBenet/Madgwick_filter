import datetime
import math
import os

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import pyIGRF
import pyproj
import scipy
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

from scipy.spatial.transform import Rotation


def load_parquet_data(path_folder, path_calibration=None):
    ts_gnss, gnss_x, gnss_y, gnss_z, gnss_lon, gnss_lat = load_parquet_gnss(os.path.join(path_folder, "gnss.parquet"))
    ts_imu, ax, ay, az, vrx, vry, vrz = load_parquet_imu(os.path.join(path_folder, "imu.parquet"))
    ts_mag, mxlf, mylf, mzlf, mxla, myla, mzla, mxra, myra, mzra, mxrf, myrf, mzrf = load_parquet_mag(
        os.path.join(path_folder, "left_forearm.parquet"),
        os.path.join(path_folder, "left_arm.parquet"),
        os.path.join(path_folder, "right_arm.parquet"),
        os.path.join(path_folder, "right_forearm.parquet"),
    )
    if path_calibration is not None:
        (
            ts_gnss_calib, gnss_x_calib, gnss_y_calib, gnss_z_calib, gnss_lon_calib, gnss_lat_calib,
            ts_imu_calib, ax_calib, ay_calib, az_calib, vrx_calib, vry_calib, vrz_calib,
            ts_mag_calib, mxlf_calib, mylf_calib, mzlf_calib, mxla_calib, myla_calib, mzla_calib, mxra_calib,
            myra_calib, mzra_calib, mxrf_calib, myrf_calib, mzrf_calib
        ) = load_parquet_data(path_calibration)

        lf_calib, la_calib, ra_calib, rf_calib = calibrate(
            path_calibration,
            gnss_lon_calib, gnss_lat_calib, gnss_z_calib,
            mxlf_calib, mylf_calib, mzlf_calib,
            mxla_calib, myla_calib, mzla_calib,
            mxra_calib, myra_calib, mzra_calib,
            mxrf_calib, myrf_calib, mzrf_calib,
        )

        print(
            f"Calib LF: {numpy.std(numpy.linalg.norm(apply_calibration(lf_calib, mxlf_calib, mylf_calib, mzlf_calib), axis=0))}")
        print(
            f"Calib LA: {numpy.std(numpy.linalg.norm(apply_calibration(la_calib, mxla_calib, myla_calib, mzla_calib), axis=0))}")
        print(
            f"Calib RA: {numpy.std(numpy.linalg.norm(apply_calibration(ra_calib, mxra_calib, myra_calib, mzra_calib), axis=0))}")
        print(
            f"Calib RF: {numpy.std(numpy.linalg.norm(apply_calibration(rf_calib, mxrf_calib, myrf_calib, mzrf_calib), axis=0))}")

        mxlf, mylf, mzlf = apply_calibration(lf_calib, mxlf, mylf, mzlf)
        mxla, myla, mzla = apply_calibration(la_calib, mxla, myla, mzla)
        mxra, myra, mzra = apply_calibration(ra_calib, mxra, myra, mzra)
        mxrf, myrf, mzrf = apply_calibration(rf_calib, mxrf, myrf, mzrf)

    return (
        ts_gnss, gnss_x, gnss_y, gnss_z, gnss_lon, gnss_lat,
        ts_imu, ax, ay, az, vrx, vry, vrz,
        ts_mag, mxlf, mylf, mzlf, mxla, myla, mzla, mxra, myra, mzra, mxrf, myrf, mzrf
    )


def load_parquet_gnss(path_gnss):
    gnss_data = pd.read_parquet(path_gnss, engine="pyarrow")
    lat = gnss_data["lat"].values
    lon = gnss_data["lon"].values
    alt = gnss_data["alt"].values
    gnss_ts = gnss_data["timestamps"].values
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return gnss_ts, x, y, alt, lon, lat


def load_parquet_imu(path_imu):
    imu_data = pd.read_parquet(path_imu, engine="pyarrow")
    imu_ts = imu_data["timestamps"].values
    acc_value = imu_data.iloc[:, 1:4].values
    gyr_value = imu_data.iloc[:, 4:7].values * np.pi / 180
    ax, ay, az = acc_value[:, 0], acc_value[:, 1], acc_value[:, 2]
    vrx, vry, vrz = gyr_value[:, 0], gyr_value[:, 1], gyr_value[:, 2]
    return imu_ts, ax, ay, az, vrx, vry, vrz


def load_parquet_mag(path_left_forearm, path_left_arm, path_right_arm, path_right_forearm):
    left_forearm_data = pd.read_parquet(path_left_forearm, engine="pyarrow")
    left_arm_data = pd.read_parquet(path_left_arm, engine="pyarrow")
    right_arm_data = pd.read_parquet(path_right_arm, engine="pyarrow")
    right_forearm_data = pd.read_parquet(path_right_forearm, engine="pyarrow")
    mag_left_side = np.c_[left_forearm_data.iloc[:, 1:].values, left_arm_data.iloc[:, 1:].values]
    mag_right_side = np.c_[right_arm_data.iloc[:, 1:].values, right_forearm_data.iloc[:, 1:].values,]
    ts_start = max(left_forearm_data.iloc[0, 0], right_forearm_data.iloc[0, 0])
    ts_end = min(left_forearm_data.iloc[-1, 0], right_forearm_data.iloc[-1, 0])
    start_mag_ts_left = np.searchsorted(left_forearm_data.iloc[0], ts_start)
    start_mag_ts_right = np.searchsorted(right_forearm_data.iloc[:, 0], ts_start)
    end_mag_ts_right = np.searchsorted(right_forearm_data.iloc[:, 0], ts_end)
    mag_right_side = mag_right_side[start_mag_ts_right:end_mag_ts_right]
    ts_mag = right_forearm_data.iloc[start_mag_ts_right:end_mag_ts_right, 0].values
    mag_left_temp = mag_left_side[start_mag_ts_left:len(mag_right_side) + start_mag_ts_left]
    if len(mag_left_temp) < len(mag_right_side):
        mag_right_side = mag_right_side[:len(mag_left_temp)]
        ts_mag = ts_mag[:len(mag_left_temp)]
    mag_data = np.c_[mag_left_side[start_mag_ts_left:len(mag_right_side) + start_mag_ts_left], mag_right_side]
    mxlf = mag_data[:, 0]
    mylf = mag_data[:, 1]
    mzlf = mag_data[:, 2]
    mxla = mag_data[:, 3]
    myla = mag_data[:, 4]
    mzla = mag_data[:, 5]
    mxra = mag_data[:, 6]
    myra = mag_data[:, 7]
    mzra = mag_data[:, 8]
    mxrf = mag_data[:, 9]
    myrf = mag_data[:, 10]
    mzrf = mag_data[:, 11]
    return ts_mag, mxlf, mylf, mzlf, mxla, myla, mzla, mxra, myra, mzra, mxrf, myrf, mzrf


def get_mag_vect(lon, lat, alt, path_calibration):
    split = path_calibration.split(os.sep)[-1]
    split_2 = split.split("T")[0]
    year, month, day = split_2.split("-")
    year = int(year)
    month = int(month)
    day = int(day)
    date = datetime.datetime(year=year, month=month, day=day)
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    days_in_year = 366 if is_leap else 365
    year_start = datetime.datetime(year, 1, 1)
    days_elapsed = (date - year_start).total_seconds() / (24 * 3600)
    decimal_year = year + (days_elapsed / days_in_year)
    avg_lat = np.mean(lat)
    avg_lon = np.mean(lon)
    avg_alt = np.mean(alt)
    mag_vect = pyIGRF.igrf_value(avg_lat, avg_lon, avg_alt, decimal_year)[-3:]
    return mag_vect


def moving_average(a, n=3):
    return uniform_filter1d(a, size=n, mode='reflect')[n:-n]


def get_enveloppe(ts, mag_signal_1d, sliding_time=2):
    freq = 1 / np.median(np.diff(ts, ))
    n = int(sliding_time * freq + 1)
    enveloppe_max = maximum_filter1d(mag_signal_1d, size=n, mode='reflect')
    enveloppe_min = minimum_filter1d(mag_signal_1d, size=n, mode='reflect')
    return enveloppe_max - enveloppe_min


def interpolate_mag_on_imu(ts_imu, ts_mag, mxlf, mylf, mzlf, mxla, myla, mzla, mxra, myra, mzra, mxrf, myrf, mzrf, ax,
                           ay, az, vrx, vry, vrz):
    freq_mag = 1 / np.median(np.diff(ts_mag))
    freq_imu = 1 / np.median(np.diff(ts_imu))
    n_window = max(3, int((freq_mag / freq_imu) * 4 + 0.5))
    ts_to_interp = ts_mag[n_window:-n_window]

    mxlf_interp = scipy.interpolate.Akima1DInterpolator(ts_to_interp, moving_average(mxlf, n=n_window))(ts_imu)
    mylf_interp = scipy.interpolate.Akima1DInterpolator(ts_to_interp, moving_average(mylf, n=n_window))(ts_imu)
    mzlf_interp = scipy.interpolate.Akima1DInterpolator(ts_to_interp, moving_average(mzlf, n=n_window))(ts_imu)
    mxla_interp = scipy.interpolate.Akima1DInterpolator(ts_to_interp, moving_average(mxla, n=n_window))(ts_imu)
    myla_interp = scipy.interpolate.Akima1DInterpolator(ts_to_interp, moving_average(myla, n=n_window))(ts_imu)
    mzla_interp = scipy.interpolate.Akima1DInterpolator(ts_to_interp, moving_average(mzla, n=n_window))(ts_imu)

    mxra_interp = scipy.interpolate.Akima1DInterpolator(ts_to_interp, moving_average(mxra, n=n_window))(ts_imu)
    myra_interp = scipy.interpolate.Akima1DInterpolator(ts_to_interp, moving_average(myra, n=n_window))(ts_imu)
    mzra_interp = scipy.interpolate.Akima1DInterpolator(ts_to_interp, moving_average(mzra, n=n_window))(ts_imu)

    mxrf_interp = scipy.interpolate.Akima1DInterpolator(ts_to_interp, moving_average(mxrf, n=n_window))(ts_imu)
    myrf_interp = scipy.interpolate.Akima1DInterpolator(ts_to_interp, moving_average(myrf, n=n_window))(ts_imu)
    mzrf_interp = scipy.interpolate.Akima1DInterpolator(ts_to_interp, moving_average(mzrf, n=n_window))(ts_imu)

    ax_interp = ax
    ay_interp = ay
    az_interp = az
    vrx_interp = vrx
    vry_interp = vry
    vrz_interp = vrz

    ts = ts_imu

    res = numpy.array([
        mxlf_interp, mylf_interp, mzlf_interp,
        mxla_interp, myla_interp, mzla_interp,
        mxra_interp, myra_interp, mzra_interp,
        mxrf_interp, myrf_interp, mzrf_interp,
        ax_interp, ay_interp, az_interp,
        vrx_interp, vry_interp, vrz_interp, ts
    ])

    return res[:, ~np.isnan(res).any(axis=0)]


def interpolate_imu_on_mag(
        ts_imu, ts_mag, mxlf, mylf, mzlf, mxla, myla, mzla, mxra, myra, mzra, mxrf, myrf, mzrf, ax, ay, az, vrx, vry,
        vrz
):

    ax_interp = scipy.interpolate.Akima1DInterpolator(ts_imu, ax)(ts_mag)
    ay_interp = scipy.interpolate.Akima1DInterpolator(ts_imu, ay)(ts_mag)
    az_interp = scipy.interpolate.Akima1DInterpolator(ts_imu, az)(ts_mag)

    vrx_interp = scipy.interpolate.Akima1DInterpolator(ts_imu, vrx)(ts_mag)
    vry_interp = scipy.interpolate.Akima1DInterpolator(ts_imu, vry)(ts_mag)
    vrz_interp = scipy.interpolate.Akima1DInterpolator(ts_imu, vrz)(ts_mag)

    mxlf_interp = mxlf
    mylf_interp = mylf
    mzlf_interp = mzlf

    mxla_interp = mxla
    myla_interp = myla
    mzla_interp = mzla

    mxra_interp = mxra
    myra_interp = myra
    mzra_interp = mzra

    mxrf_interp = mxrf
    myrf_interp = myrf
    mzrf_interp = mzrf

    ts = ts_mag

    res = numpy.array([
        mxlf_interp, mylf_interp, mzlf_interp,
        mxla_interp, myla_interp, mzla_interp,
        mxra_interp, myra_interp, mzra_interp,
        mxrf_interp, myrf_interp, mzrf_interp,
        ax_interp, ay_interp, az_interp,
        vrx_interp, vry_interp, vrz_interp, ts
    ])

    return res[:, ~np.isnan(res).any(axis=0)]


def merge_mag(mxlf, mylf, mzlf, mxla, myla, mzla, mxra, myra, mzra, mxrf, myrf, mzrf):
    # mx = numpy.mean([mxlf, mxla, mxra, mxrf], axis=0)
    # my = numpy.mean([mylf, myla, myra, myrf], axis=0)
    # mz = numpy.mean([mzlf, mzla, mzra, mzrf], axis=0)
    # return mx, my, mz
    return mxlf, mylf, mzlf


def rotate_mag(mx, my, mz, quaternions):
    qw, qx, qy, qz = quaternions.T
    mx_rot = np.zeros_like(mx)
    my_rot = np.zeros_like(my)
    mz_rot = np.zeros_like(mz)
    for i in range(len(mx)):
        w, x, y, z = qw[i], qx[i], qy[i], qz[i]
        mx_rot[i] = (1 - 2 * y * y - 2 * z * z) * mx[i] + (2 * x * y - 2 * w * z) * my[i] + (
                2 * x * z + 2 * w * y) * \
                    mz[i]
        my_rot[i] = (2 * x * y + 2 * w * z) * mx[i] + (1 - 2 * x * x - 2 * z * z) * my[i] + (
                2 * y * z - 2 * w * x) * \
                    mz[i]
        mz_rot[i] = (2 * x * z - 2 * w * y) * mx[i] + (2 * y * z + 2 * w * x) * my[i] + (
                1 - 2 * x * x - 2 * y * y) * \
                    mz[i]
    return mx_rot, my_rot, mz_rot


def process_result(
        quaternions,
        mxlf_ned, mylf_ned, mzlf_ned,
        mxla_ned, myla_ned, mzla_ned,
        mxra_ned, myra_ned, mzra_ned,
        mxrf_ned, myrf_ned, mzrf_ned,
        ts_gnss, gnss_x, gnss_y, gnss_z,
        ax_ned, ay_ned, az_ned, ts_imu, ts_mag,
        mxlf, mylf, mzlf,
        mxla, myla, mzla,
        mxra, myra, mzra,
        mxrf, myrf, mzrf

):
    std_mxlf_ned = numpy.std(mxlf_ned)
    std_mylf_ned = numpy.std(mylf_ned)
    std_mzlf_ned = numpy.std(mzlf_ned)
    std_mxla_ned = numpy.std(mxla_ned)
    std_myla_ned = numpy.std(myla_ned)
    std_mzla_ned = numpy.std(mzla_ned)
    std_mxra_ned = numpy.std(mxra_ned)
    std_myra_ned = numpy.std(myra_ned)
    std_mzra_ned = numpy.std(mzra_ned)
    std_mxrf_ned = numpy.std(mxrf_ned)
    std_myrf_ned = numpy.std(myrf_ned)
    std_mzrf_ned = numpy.std(mzrf_ned)

    std_sum = (
            (std_mxlf_ned + std_mylf_ned + std_mzlf_ned) +
            (std_mxla_ned + std_myla_ned + std_mzla_ned) +
            (std_mxra_ned + std_myra_ned + std_mzra_ned) +
            (std_mxrf_ned + std_myrf_ned + std_mzrf_ned)
    )

    print(f"{std_sum = }")


def jacobian(q, g):
    gq0 = q[1] * g[0] + q[2] * g[1] + q[3] * g[2]
    gq1 = q[0] * g[0] + q[3] * g[1] - q[2] * g[2]
    gq2 = -q[3] * g[0] + q[0] * g[1] + q[1] * g[2]
    gq3 = q[2] * g[0] - q[1] * g[1] + q[0] * g[2]
    return 2 * np.array(
        [[gq1, gq2, gq3], [gq0, gq3, -gq2], [-gq3, gq0, gq1], [gq2, -gq1, gq0]]
    )


def obj_func(q, a, g):
    q01 = q[0] * q[1]
    q02 = q[0] * q[2]
    q03 = q[0] * q[3]
    q12 = q[1] * q[2]
    q13 = q[1] * q[3]
    q23 = q[2] * q[3]
    q11 = q[1] ** 2
    q22 = q[2] ** 2
    q33 = q[3] ** 2
    return np.array(
        [
            g[0]
            - 2 * (g[0] * (q22 + q33) - g[1] * (q03 + q12) - g[2] * (q13 - q02))
            - a[0],
            g[1]
            + 2 * (g[0] * (q12 - q03) - g[1] * (q11 + q33) + g[2] * (q01 + q23))
            - a[1],
            g[2]
            + 2 * (g[0] * (q02 + q13) + g[1] * (q23 - q01) - g[2] * (q11 + q22))
            - a[2],
        ],
    )


def step_vector_ref(quaternion_previous, vect_measure, vect_reference):
    vect_measure_normalized = vect_measure / np.linalg.norm(vect_measure, 2)
    step_a = np.dot(
        jacobian(quaternion_previous, vect_reference),
        obj_func(quaternion_previous, vect_measure_normalized, vect_reference)
    )
    norm = np.linalg.norm(step_a, 2)
    return step_a / norm if norm > 0 else step_a


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


def madgwick(
    timestamps, acc, gyr, mag,
    gain_acc, gain_mag, acc_weight, mag_weight,
    earth_vector, q0=None,
    forward=True
):
    n = len(timestamps)
    quaternions = np.zeros((n, 4))

    if forward:
        start, end, step = 1, n, 1
        initial_index = 0
        get_dt = lambda _i: timestamps[_i] - timestamps[_i - 1]
        prev_q = lambda _i: quaternions[_i - 1]
        coef = lambda _i: n - _i if q0 is None else 1
    else:
        start, end, step = n - 2, -1, -1
        initial_index = n - 1
        get_dt = lambda _i: timestamps[_i + 1] - timestamps[_i]
        prev_q = lambda _i: quaternions[_i + 1]
        coef = lambda _i: _i + 1 if q0 is None else 1

    quaternions[initial_index] = numpy.array([1, 0, 0, 0]) if q0 is None else q0

    a = []
    b = []

    for i in range(start, end, step):
        step_acc = step_vector_ref(prev_q(i), acc[i], np.array([0, 0, 1]))
        step_mag = step_vector_ref(prev_q(i), mag[i], earth_vector)
        step_gyr = product_quaternion(prev_q(i), step * gyr[i])

        # q_no_yaw = remove_yaw_from_quaternion(prev_q(i))
        # step_acc_no_yaw = step_vector_ref(q_no_yaw, acc[i], np.array([0, 0, 1]))
        # norm_acc_no_yaw = np.linalg.norm(step_acc_no_yaw)
        # norm_mag_no_yaw = np.linalg.norm(step_mag)
        # step_mag_n = step_mag / norm_mag_no_yaw
        # step_acc_n = step_acc_no_yaw / norm_acc_no_yaw
        # cos_acc_mag = np.dot(step_mag_n, step_acc_n)
        # a.append(cos_acc_mag)
        # w_cos_acc_mag = ((cos_acc_mag + 1) / 2)**0.1
        # b.append(w_cos_acc_mag)
        # step_total = 0.5 * step_gyr - (gain_acc * step_acc * acc_weight[i] + gain_mag * step_mag * mag_weight[i]) * coef(i) * w_cos_acc_mag

        step_total = 0.5 * step_gyr - (gain_acc * step_acc + gain_mag * step_mag) * coef(i)
        quaternions[i] = normalize_quaternion(prev_q(i) + get_dt(i) * step_total)
    return normalize_quaternion(quaternions)


def remove_yaw_from_quaternion(q):
    yaw, pitch, roll = quaternion_to_euler(q)
    q_no_yaw = euler_to_quaternion(0, pitch, roll)
    return q_no_yaw


def product_quaternion(q1, gyro):
    return np.array([
        -q1[1] * gyro[0] - q1[2] * gyro[1] - q1[3] * gyro[2],
        q1[0] * gyro[0] + q1[2] * gyro[2] - q1[3] * gyro[1],
        q1[0] * gyro[1] - q1[1] * gyro[2] + q1[3] * gyro[0],
        q1[0] * gyro[2] + q1[1] * gyro[1] - q1[2] * gyro[0],
    ])


def calculate_adaptive_weight(n, thres=0.2):
    return (1 - 2 * thres) * (numpy.array([i for i in range(n)]) / (n - 1)) + thres


def calibrate(
        path_calibration,
        gnss_lon_calib, gnss_lat_calib, gnss_z_calib,
        mxlf_calib, mylf_calib, mzlf_calib,
        mxla_calib, myla_calib, mzla_calib,
        mxra_calib, myra_calib, mzra_calib,
        mxrf_calib, myrf_calib, mzrf_calib,
):
    mag_ter = get_mag_vect(gnss_lon_calib, gnss_lat_calib, gnss_z_calib, path_calibration)
    lf_calib = calcul_calibration(mxlf_calib, mylf_calib, mzlf_calib, mag_ter)
    la_calib = calcul_calibration(mxla_calib, myla_calib, mzla_calib, mag_ter)
    ra_calib = calcul_calibration(mxra_calib, myra_calib, mzra_calib, mag_ter)
    rf_calib = calcul_calibration(mxrf_calib, myrf_calib, mzrf_calib, mag_ter)
    return lf_calib, la_calib, ra_calib, rf_calib


def apply_calibration(calibration, mx, my, mz):
    p, b = calibration
    mag = numpy.array([mx, my, mz]).T - b
    return np.dot(p, mag.T)


def calcul_calibration(mx, my, mz, mag_ter):
    mag = numpy.array([mx, my, mz]).T
    D = np.c_[
        mag ** 2,
        mag[:, :1] * mag[:, 1:2],
        mag[:, :1] * mag[:, 2:],
        mag[:, 1:2] * mag[:, 2:],
        mag[:, :1],
        mag[:, 1:2],
        mag[:, 2:]]
    E = np.dot(D.T, D)
    a = np.dot(np.dot(np.linalg.inv(E), D.T), np.ones(len(mag)))
    n = np.array([[a[0], a[3] / 2, a[4] / 2], [a[3] / 2, a[1], a[5] / 2], [a[4] / 2, a[5] / 2, a[2]]])
    v, u = np.linalg.eig(n)
    v = np.diag(v)
    Q = np.zeros(np.shape(u))
    for i in range(3):
        Q[:, i] = u[:, i] / np.linalg.norm(u[:, i])
    m = np.linalg.norm(mag_ter) * np.dot(np.dot(Q, np.sqrt(v)), Q.T)
    b = np.dot(-0.5 * a[6:9], np.linalg.pinv(n))
    return m, b


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


def quaternion_to_euler(quaternions):
    qw, qx, qy, qz = quaternions.T

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx ** 2 + qy ** 2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))  # Clip for numerical stability

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy ** 2 + qz ** 2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    yaw *= 180 / math.pi
    pitch *= 180 / math.pi
    pitch *= 180 / math.pi

    return yaw, pitch, roll


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
    diff_x = numpy.diff(x_gnss)
    diff_y = numpy.diff(y_gnss)
    res = -numpy.arctan2(diff_y, diff_x) * 180 / numpy.pi
    res = uniform_filter1d(res, size=int(duration_gnss_filter / np.median(np.diff(ts_gnss))), axis=0, mode="reflect")
    return res


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


def low_pass_quaternion_filter(quaternions, window_size=5):
    N = len(quaternions)
    quaternions = np.asarray(quaternions)
    quaternions_low_pass = np.zeros_like(quaternions)
    for i in range(N):
        start = max(0, i - window_size + 1)
        window = quaternions[start:i + 1]
        if len(window) == 1:
            quaternions_low_pass[i] = window[0]
        else:
            quaternions_low_pass[i] = average_quaternions(window)
    return quaternions_low_pass


def align_quaternions(reference, quats):
    aligned = []
    for q in quats:
        if np.dot(reference, q) < 0:
            aligned.append(-q)
        else:
            aligned.append(q)
    return np.array(aligned)


def average_quaternions(quats):
    # Filtrer quaternions de norme nulle ou trop petite
    norms = np.linalg.norm(quats, axis=1)
    valid_quats = quats[norms > 1e-8]
    if len(valid_quats) == 0:
        raise ValueError("Aucune quaternion valide dans la fenêtre.")

    valid_quats = valid_quats / np.linalg.norm(valid_quats, axis=1, keepdims=True)
    valid_quats = align_quaternions(valid_quats[0], valid_quats)
    rot_mats = Rotation.from_quat(valid_quats).as_matrix()
    avg_mat = np.mean(rot_mats, axis=0)
    u, _, vt = np.linalg.svd(avg_mat)
    r_avg = u @ vt
    return Rotation.from_matrix(r_avg).as_quat()


def quaternion_distance(q1, q2):
    """Angular distance between two quaternions."""
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return 2 * np.arccos(dot_product)
