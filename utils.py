import datetime
import os
import numpy
import pandas
import pyIGRF
import pyproj
import scipy
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

import rotations
from pyqgis_tools import point_cloud


def load_parquet_data(path_folder, path_calibration=None, path_imu=None):
    ts_gnss, gnss_x, gnss_y, gnss_z, gnss_lon, gnss_lat, epsg = load_parquet_gnss(path_folder)
    ts_imu, ax, ay, az, vrx, vry, vrz = load_parquet_imu(path_folder, path_imu)
    ts_mag, mxlf, mylf, mzlf, mxla, myla, mzla, mxra, myra, mzra, mxrf, myrf, mzrf = load_parquet_mag(
        os.path.join(path_folder, "left_forearm.parquet"),
        os.path.join(path_folder, "left_arm.parquet"),
        os.path.join(path_folder, "right_arm.parquet"),
        os.path.join(path_folder, "right_forearm.parquet"),
    )

    if path_calibration is not None:
        (
            ts_gnss_calib, gnss_x_calib, gnss_y_calib, gnss_z_calib, gnss_lon_calib, gnss_lat_calib, epsg,
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
            f"Calib LF: {round(numpy.std(numpy.linalg.norm(apply_calibration(lf_calib, mxlf_calib, mylf_calib, mzlf_calib), axis=0)), 1)}nT")
        print(
            f"Calib LA: {round(numpy.std(numpy.linalg.norm(apply_calibration(la_calib, mxla_calib, myla_calib, mzla_calib), axis=0)), 1)}nT")
        print(
            f"Calib RA: {round(numpy.std(numpy.linalg.norm(apply_calibration(ra_calib, mxra_calib, myra_calib, mzra_calib), axis=0)), 1)}nT")
        print(
            f"Calib RF: {round(numpy.std(numpy.linalg.norm(apply_calibration(rf_calib, mxrf_calib, myrf_calib, mzrf_calib), axis=0)), 1)}nT")

        mxlf, mylf, mzlf = apply_calibration(lf_calib, mxlf, mylf, mzlf)
        mxla, myla, mzla = apply_calibration(la_calib, mxla, myla, mzla)
        mxra, myra, mzra = apply_calibration(ra_calib, mxra, myra, mzra)
        mxrf, myrf, mzrf = apply_calibration(rf_calib, mxrf, myrf, mzrf)

    return (
        ts_gnss, gnss_x, gnss_y, gnss_z, gnss_lon, gnss_lat, epsg,
        ts_imu, ax, ay, az, vrx, vry, vrz,
        ts_mag, mxlf, mylf, mzlf, mxla, myla, mzla, mxra, myra, mzra, mxrf, myrf, mzrf
    )


def load_parquet_gnss(path_gnss):
    gnss_data = pandas.read_parquet(os.path.join(path_gnss, "gnss.parquet"), engine="pyarrow")
    lat = gnss_data["lat"].values
    lon = gnss_data["lon"].values
    alt = gnss_data["alt"].values
    gnss_ts = gnss_data["timestamps"].values
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    # TODO: Mettre un EPSG dynamique UTM
    x, y = transformer.transform(lon, lat)
    return gnss_ts, x, y, alt, lon, lat, 32631


def load_parquet_imu(path_imu, path_calib_imu=None):
    imu_data = pandas.read_parquet(os.path.join(path_imu, "imu.parquet"), engine="pyarrow")
    imu_ts = imu_data["timestamps"].values
    acc_value = imu_data.iloc[:, 1:4].values
    gyr_value = imu_data.iloc[:, 4:7].values * numpy.pi / 180
    ax, ay, az = acc_value[:, 0], acc_value[:, 1], acc_value[:, 2]
    vrx, vry, vrz = gyr_value[:, 0], gyr_value[:, 1], gyr_value[:, 2]
    if path_calib_imu is not None:
        ts_imu_calib, ax_calib, ay_calib, az_calib, vrx_calib, vry_calib, vrz_calib = load_parquet_imu(path_calib_imu)
        ax_lp = low_pass(ax_calib, ts_imu_calib, t=1)
        ay_lp = low_pass(ay_calib, ts_imu_calib, t=1)
        az_lp = low_pass(az_calib, ts_imu_calib, t=1)
        jx = numpy.diff(ax_lp) / numpy.diff(ts_imu_calib)
        jy = numpy.diff(ay_lp) / numpy.diff(ts_imu_calib)
        jz = numpy.diff(az_lp) / numpy.diff(ts_imu_calib)
        jn = numpy.linalg.norm([jx, jy, jz], axis=0)
        mask = jn < numpy.percentile(jn, 10)
        ax_filtered = ax_lp[1:][mask]
        ay_filtered = ay_lp[1:][mask]
        az_filtered = az_lp[1:][mask]
        m, b = calcul_calibration(ax_filtered, ay_filtered, az_filtered, numpy.array([0, 0, 1]))
        ax_calibrated, ay_calibrated, az_calibrated = apply_calibration(
            (m, b), ax_filtered, ay_filtered, az_filtered)
        norm_uncalibrated = numpy.linalg.norm([ax_filtered, ay_filtered, az_filtered], axis=0)
        norm_calibrated = numpy.linalg.norm([ax_calibrated, ay_calibrated, az_calibrated], axis=0)

        std_uncalibrated = numpy.std(norm_uncalibrated)
        std_calibrated = numpy.std(norm_calibrated)

        print(f"{std_uncalibrated = }")
        print(f"{std_calibrated = }")

        print(f"Calib IMU: Improvment x {round(std_uncalibrated / std_calibrated, 2)}")
        ax, ay, az = apply_calibration((m, b), ax, ay, az)

    return imu_ts, ax, ay, az, vrx, vry, vrz


def load_parquet_mag(path_left_forearm, path_left_arm, path_right_arm, path_right_forearm):
    left_forearm_data = pandas.read_parquet(path_left_forearm, engine="pyarrow")
    left_arm_data = pandas.read_parquet(path_left_arm, engine="pyarrow")
    right_arm_data = pandas.read_parquet(path_right_arm, engine="pyarrow")
    right_forearm_data = pandas.read_parquet(path_right_forearm, engine="pyarrow")
    mag_left_side = numpy.c_[left_forearm_data.iloc[:, 1:].values, left_arm_data.iloc[:, 1:].values]
    mag_right_side = numpy.c_[right_arm_data.iloc[:, 1:].values, right_forearm_data.iloc[:, 1:].values,]
    ts_start = max(left_forearm_data.iloc[0, 0], right_forearm_data.iloc[0, 0])
    ts_end = min(left_forearm_data.iloc[-1, 0], right_forearm_data.iloc[-1, 0])
    start_mag_ts_left = numpy.searchsorted(left_forearm_data.iloc[0], ts_start)
    start_mag_ts_right = numpy.searchsorted(right_forearm_data.iloc[:, 0], ts_start)
    end_mag_ts_right = numpy.searchsorted(right_forearm_data.iloc[:, 0], ts_end)
    mag_right_side = mag_right_side[start_mag_ts_right:end_mag_ts_right]
    ts_mag = right_forearm_data.iloc[start_mag_ts_right:end_mag_ts_right, 0].values
    mag_left_temp = mag_left_side[start_mag_ts_left:len(mag_right_side) + start_mag_ts_left]
    if len(mag_left_temp) < len(mag_right_side):
        mag_right_side = mag_right_side[:len(mag_left_temp)]
        ts_mag = ts_mag[:len(mag_left_temp)]
    mag_data = numpy.c_[mag_left_side[start_mag_ts_left:len(mag_right_side) + start_mag_ts_left], mag_right_side]
    return ts_mag, *mag_data.T


def get_mag_vect(lon, lat, alt, path_acquisition):
    d, i, h, x, y, z, f = pyIGRF.igrf_value(
        lat=numpy.mean(lat),
        lon=numpy.mean(lon),
        alt=numpy.mean(alt),
        year=path_to_decimal_year(path_acquisition)
    )
    return x, y, z  # NED mag vect


def path_to_decimal_year(path):
    split = path.split(os.sep)[-1]
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
    return decimal_year


def get_enveloppe(ts, mag_signal_1d, sliding_time=2):
    freq = 1 / numpy.median(numpy.diff(ts, ))
    n = int(sliding_time * freq + 1)
    enveloppe_max = maximum_filter1d(mag_signal_1d, size=n, mode='reflect')
    enveloppe_min = minimum_filter1d(mag_signal_1d, size=n, mode='reflect')
    return enveloppe_max - enveloppe_min


def interpolate_mag_on_imu(
    ts_imu, ts_mag,
    mxlf, mylf, mzlf, mxla, myla, mzla, mxra, myra, mzra, mxrf, myrf, mzrf,
    ax, ay, az, vrx, vry, vrz,
    duration_filter_mag_axis
):
    n_window = int(duration_filter_mag_axis / numpy.median(numpy.diff(ts_mag)))

    mxlf_interp = scipy.interpolate.Akima1DInterpolator(ts_mag, low_pass(mxlf, ts=None, n=n_window, t=None))(ts_imu)
    mylf_interp = scipy.interpolate.Akima1DInterpolator(ts_mag, low_pass(mylf, ts=None, n=n_window, t=None))(ts_imu)
    mzlf_interp = scipy.interpolate.Akima1DInterpolator(ts_mag, low_pass(mzlf, ts=None, n=n_window, t=None))(ts_imu)
    mxla_interp = scipy.interpolate.Akima1DInterpolator(ts_mag, low_pass(mxla, ts=None, n=n_window, t=None))(ts_imu)
    myla_interp = scipy.interpolate.Akima1DInterpolator(ts_mag, low_pass(myla, ts=None, n=n_window, t=None))(ts_imu)
    mzla_interp = scipy.interpolate.Akima1DInterpolator(ts_mag, low_pass(mzla, ts=None, n=n_window, t=None))(ts_imu)
    mxra_interp = scipy.interpolate.Akima1DInterpolator(ts_mag, low_pass(mxra, ts=None, n=n_window, t=None))(ts_imu)
    myra_interp = scipy.interpolate.Akima1DInterpolator(ts_mag, low_pass(myra, ts=None, n=n_window, t=None))(ts_imu)
    mzra_interp = scipy.interpolate.Akima1DInterpolator(ts_mag, low_pass(mzra, ts=None, n=n_window, t=None))(ts_imu)
    mxrf_interp = scipy.interpolate.Akima1DInterpolator(ts_mag, low_pass(mxrf, ts=None, n=n_window, t=None))(ts_imu)
    myrf_interp = scipy.interpolate.Akima1DInterpolator(ts_mag, low_pass(myrf, ts=None, n=n_window, t=None))(ts_imu)
    mzrf_interp = scipy.interpolate.Akima1DInterpolator(ts_mag, low_pass(mzrf, ts=None, n=n_window, t=None))(ts_imu)

    res = numpy.array([
        mxlf_interp, mylf_interp, mzlf_interp,
        mxla_interp, myla_interp, mzla_interp,
        mxra_interp, myra_interp, mzra_interp,
        mxrf_interp, myrf_interp, mzrf_interp,
        ax, ay, az,
        vrx, vry, vrz, ts_imu])
    return res[:, ~numpy.isnan(res).any(axis=0)]


def interpolate_imu_on_mag(
        ts_imu, ts_mag, mxlf, mylf, mzlf, mxla, myla, mzla, mxra, myra, mzra, mxrf, myrf, mzrf, ax, ay, az, vrx, vry,
        vrz, duration_filter_mag_axis
):

    ax_interp = scipy.interpolate.Akima1DInterpolator(ts_imu, ax)(ts_mag)
    ay_interp = scipy.interpolate.Akima1DInterpolator(ts_imu, ay)(ts_mag)
    az_interp = scipy.interpolate.Akima1DInterpolator(ts_imu, az)(ts_mag)
    vrx_interp = scipy.interpolate.Akima1DInterpolator(ts_imu, vrx)(ts_mag)
    vry_interp = scipy.interpolate.Akima1DInterpolator(ts_imu, vry)(ts_mag)
    vrz_interp = scipy.interpolate.Akima1DInterpolator(ts_imu, vrz)(ts_mag)

    mxlf_interp = low_pass(mxlf, ts=ts_mag, n=None, t=duration_filter_mag_axis)
    mylf_interp = low_pass(mylf, ts=ts_mag, n=None, t=duration_filter_mag_axis)
    mzlf_interp = low_pass(mzlf, ts=ts_mag, n=None, t=duration_filter_mag_axis)

    mxla_interp = low_pass(mxla, ts=ts_mag, n=None, t=duration_filter_mag_axis)
    myla_interp = low_pass(myla, ts=ts_mag, n=None, t=duration_filter_mag_axis)
    mzla_interp = low_pass(mzla, ts=ts_mag, n=None, t=duration_filter_mag_axis)

    mxra_interp = low_pass(mxra, ts=ts_mag, n=None, t=duration_filter_mag_axis)
    myra_interp = low_pass(myra, ts=ts_mag, n=None, t=duration_filter_mag_axis)
    mzra_interp = low_pass(mzra, ts=ts_mag, n=None, t=duration_filter_mag_axis)

    mxrf_interp = low_pass(mxrf, ts=ts_mag, n=None, t=duration_filter_mag_axis)
    myrf_interp = low_pass(myrf, ts=ts_mag, n=None, t=duration_filter_mag_axis)
    mzrf_interp = low_pass(mzrf, ts=ts_mag, n=None, t=duration_filter_mag_axis)

    res = numpy.array([
        mxlf_interp, mylf_interp, mzlf_interp,
        mxla_interp, myla_interp, mzla_interp,
        mxra_interp, myra_interp, mzra_interp,
        mxrf_interp, myrf_interp, mzrf_interp,
        ax_interp, ay_interp, az_interp,
        vrx_interp, vry_interp, vrz_interp, ts_mag
    ])

    return res[:, ~numpy.isnan(res).any(axis=0)]


def merge_mag(mxlf, mylf, mzlf, mxla, myla, mzla, mxra, myra, mzra, mxrf, myrf, mzrf):

    # fig, axs = plt.subplots(3)
    # axs[0].plot(mxlf, label="mxlf")
    # axs[1].plot(mylf, label="mylf")
    # axs[2].plot(mzlf, label="mzlf")
    # axs[0].plot(mxla, label="mxla")
    # axs[1].plot(myla, label="myla")
    # axs[2].plot(mzla, label="mzla")
    # axs[0].plot(mxra, label="mxra")
    # axs[1].plot(myra, label="myra")
    # axs[2].plot(mzra, label="mzra")
    # axs[0].plot(mxrf, label="mxrf")
    # axs[1].plot(myrf, label="myrf")
    # axs[2].plot(mzrf, label="mzrf")
    #
    # axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    # plt.show()

    mx = numpy.mean([mxlf, mxla, mxra, mxrf], axis=0)
    my = numpy.mean([mylf, myla, myra, myrf], axis=0)
    mz = numpy.mean([mzlf, mzla, mzra, mzrf], axis=0)
    return mx, my, mz
    # return mxlf, mylf, mzlf


def madgwick(
    timestamps, acc, gyr, mag,
    gain_acc, gain_mag, beta,
    earth_vector, q0=None,
    forward=True,
    normalize_madgwick_step=True,
):
    n = len(timestamps)

    quaternions = numpy.zeros((n, 4))

    if forward:
        start, end, step = 1, n, 1
        initial_index = 0
        get_dt = lambda _i: timestamps[_i] - timestamps[_i - 1]
        prev_q = lambda _i: quaternions[_i - 1]
        coef = lambda _i: (n - _i)**0.5 if q0 is None else 1
    else:
        start, end, step = n - 2, -1, -1
        initial_index = n - 1
        get_dt = lambda _i: timestamps[_i + 1] - timestamps[_i]
        prev_q = lambda _i: quaternions[_i + 1]
        coef = lambda _i: (_i + 1)**0.5 if q0 is None else 1

    quaternions[initial_index] = numpy.array([1, 0, 0, 0]) if q0 is None else q0
    for i in range(start, end, step):
        quaternions[i] = rotations.madgwick_step(
            prev_q(i), get_dt(i), coef(i),
            mag[i], gyr[i], acc[i],
            gain_acc[i], gain_mag[i], step, earth_vector, beta[i], normalize_madgwick_step)
    return quaternions


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
    return numpy.dot(p, mag.T)


def calcul_calibration(mx, my, mz, mag_ter):
    mag = numpy.array([mx, my, mz]).T
    D = numpy.c_[
        mag ** 2,
        mag[:, :1] * mag[:, 1:2],
        mag[:, :1] * mag[:, 2:],
        mag[:, 1:2] * mag[:, 2:],
        mag[:, :1],
        mag[:, 1:2],
        mag[:, 2:]]
    E = numpy.dot(D.T, D)
    a = numpy.dot(numpy.dot(numpy.linalg.inv(E), D.T), numpy.ones(len(mag)))
    n = numpy.array([[a[0], a[3] / 2, a[4] / 2], [a[3] / 2, a[1], a[5] / 2], [a[4] / 2, a[5] / 2, a[2]]])
    v, u = numpy.linalg.eig(n)
    v = numpy.diag(v)
    Q = numpy.zeros(numpy.shape(u))
    for i in range(3):
        Q[:, i] = u[:, i] / numpy.linalg.norm(u[:, i])
    m = numpy.linalg.norm(mag_ter) * numpy.dot(numpy.dot(Q, numpy.sqrt(v)), Q.T)
    b = numpy.dot(-0.5 * a[6:9], numpy.linalg.pinv(n))
    return m, b


def normalize(ax_interp, ay_interp, az_interp, vrx_interp, vry_interp, vrz_interp, mx_interp, my_interp, mz_interp, mag_vect_raw):
    acc_raw = numpy.column_stack((ax_interp, ay_interp, az_interp))
    mag_raw = numpy.array([mx_interp, my_interp, mz_interp]).T
    gyr = numpy.column_stack((vrx_interp, vry_interp, vrz_interp))

    mag_vect_norm = numpy.linalg.norm(mag_vect_raw)
    mag_norm = numpy.linalg.norm(mag_raw, axis=1)
    acc_norm = numpy.linalg.norm(acc_raw, axis=1)
    gyr_norm = numpy.linalg.norm(gyr, axis=1)

    mag_vect = mag_vect_raw / mag_vect_norm
    acc = numpy.column_stack((ax_interp / acc_norm, ay_interp / acc_norm, az_interp / acc_norm))
    mag = numpy.array([mx_interp / mag_norm, my_interp / mag_norm, mz_interp / mag_norm]).T
    return mag, gyr, acc, mag_vect, mag_norm, acc_norm, gyr_norm


def ahrs(
    gnss_z, gnss_lon, gnss_lat,
    ts_imu, ax, ay, az, vrx, vry, vrz,
    ts_mag, mxlf, mylf, mzlf, mxla, myla, mzla, mxra, myra, mzra, mxrf, myrf, mzrf,
    duration_filter_mag_axis, duration_filter_mag_merged, duration_filter_gyr, n_filter_acc,
    duration_filter_quaternions_outputs, gain_acc, gain_mag, path_folder, ts_mag_to_imu, ahrs_func,
    adaptative_beta, adaptative_gain_acc, adaptative_gain_mag, beta, normalize_madgwick_step
):
    if ts_mag_to_imu:
        (
            mxlf_interp, mylf_interp, mzlf_interp,
            mxla_interp, myla_interp, mzla_interp,
            mxra_interp, myra_interp, mzra_interp,
            mxrf_interp, myrf_interp, mzrf_interp,
            ax_interp, ay_interp, az_interp,
            vrx_interp, vry_interp, vrz_interp,
            ts
        ) = interpolate_mag_on_imu(
            ts_imu, ts_mag,
            mxlf, mylf, mzlf,
            mxla, myla, mzla,
            mxra, myra, mzra,
            mxrf, myrf, mzrf,
            ax, ay, az, vrx, vry, vrz, duration_filter_mag_axis)
    else:
        (
            mxlf_interp, mylf_interp, mzlf_interp,
            mxla_interp, myla_interp, mzla_interp,
            mxra_interp, myra_interp, mzra_interp,
            mxrf_interp, myrf_interp, mzrf_interp,
            ax_interp, ay_interp, az_interp,
            vrx_interp, vry_interp, vrz_interp,
            ts
        ) = interpolate_imu_on_mag(
            ts_imu, ts_mag,
            mxlf, mylf, mzlf,
            mxla, myla, mzla,
            mxra, myra, mzra,
            mxrf, myrf, mzrf,
            ax, ay, az, vrx, vry, vrz, duration_filter_mag_axis)

    mag_vect_raw = get_mag_vect(gnss_lon, gnss_lat, gnss_z, path_folder)

    mx_interp, my_interp, mz_interp = merge_mag(
        mxlf_interp, mylf_interp, mzlf_interp,
        mxla_interp, myla_interp, mzla_interp,
        mxra_interp, myra_interp, mzra_interp,
        mxrf_interp, myrf_interp, mzrf_interp)

    mag, gyr, acc, mag_vect, mag_norm, acc_norm, gyr_norm = normalize(
        ax_interp, ay_interp, az_interp,
        vrx_interp, vry_interp, vrz_interp,
        mx_interp, my_interp, mz_interp,
        mag_vect_raw)

    gains_acc, gains_mag, betas = adaptative_gains(
        adaptative_beta, adaptative_gain_acc, adaptative_gain_mag,
        gain_acc, gain_mag, beta,
        mag_norm, acc_norm, gyr_norm
    )

    quaternions = ahrs_func(
        ts,
        low_pass(mag, ts=ts, n=None, t=duration_filter_mag_merged),
        low_pass(gyr, ts=ts, n=None, t=duration_filter_gyr),
        low_pass(acc, ts=None, n=n_filter_acc, t=None),
        mag_vect,
        gain_acc=gains_acc,
        gain_mag=gains_mag,
        beta=betas,
        normalize_madgwick_step=normalize_madgwick_step,
    )
    quaternions = rotations.low_pass_quaternions(quaternions, ts, duration_filter_quaternions_outputs)
    return (
        ts, quaternions,
        mx_interp, my_interp, mz_interp,
        ax_interp, ay_interp, az_interp,
        mag_vect_raw,
        mxlf_interp, mylf_interp, mzlf_interp,
        mxla_interp, myla_interp, mzla_interp,
        mxra_interp, myra_interp, mzra_interp,
        mxrf_interp, myrf_interp, mzrf_interp,
    )


def adaptative_gains(
    adaptative_beta, adaptative_gain_acc, adaptative_gain_mag,
    gain_acc, gain_mag, beta,
    mag_norm, acc_norm, gyr_norm
):
    avg_mag = numpy.median(mag_norm)
    avg_acc = numpy.median(acc_norm)
    avg_gyr = numpy.median(gyr_norm)
    std_mag = numpy.std(mag_norm)
    std_acc = numpy.std(acc_norm)
    std_gyr = numpy.std(gyr_norm)
    error_acc = numpy.abs(acc_norm - avg_acc) + 1e-5
    error_mag = numpy.abs(mag_norm - avg_mag) + 1e-5
    error_gyr = numpy.abs(gyr_norm - avg_gyr) + 1e-5
    norm_div_std_mag = error_mag / std_mag
    norm_div_std_acc = error_acc / std_acc
    norm_div_std_gyr = error_gyr / std_gyr

    gains_acc = numpy.minimum(numpy.minimum(1, acc_norm), (1 / norm_div_std_acc))**adaptative_gain_acc * gain_acc
    gains_mag = numpy.minimum(1, (1 / norm_div_std_mag))**adaptative_gain_mag * gain_mag
    betas = (1 / ((numpy.maximum(1, norm_div_std_mag) + numpy.maximum(1, norm_div_std_acc) + numpy.maximum(1, norm_div_std_gyr)) / 3))**adaptative_beta * beta

    # plt.plot(acc_norm, label="acc_norm")
    # plt.plot(error_acc, label="error_acc")
    # plt.plot(norm_div_std_acc, label="norm_div_std_acc")
    # plt.plot(gains_acc/gain_acc, label="gains_acc")
    # plt.plot(gains_mag/gain_mag, label="gains_mag")
    # plt.plot(betas/beta, label="betas")
    # plt.legend()
    # plt.show()

    return gains_acc, gains_mag, betas


def low_pass(data, ts=None, n=None, t=None):
    if n is None:
        n = int(t / numpy.median(numpy.diff(ts)))
    if n < 2:
        return data
    return uniform_filter1d(data, size=n, axis=0, mode="reflect")


def ahrs_madgwick_python_benet(ts, mag, gyr, acc, mag_vect, gain_acc, gain_mag, beta, normalize_madgwick_step):
    n_sample = int(60 / numpy.median(numpy.diff(ts)))
    q0_backward = madgwick(
        timestamps=ts[-n_sample:],
        acc=acc[-n_sample:],
        gyr=gyr[-n_sample:],
        mag=mag[-n_sample:],
        earth_vector=mag_vect,
        q0=None,
        gain_acc=gain_acc,
        gain_mag=gain_mag,
        beta=beta,
        normalize_madgwick_step=normalize_madgwick_step,
        forward=True)[-1]
    q0_forward = madgwick(
        timestamps=ts, acc=acc, gyr=gyr, mag=mag, earth_vector=mag_vect,
        q0=q0_backward,
        gain_acc=gain_acc,
        gain_mag=gain_mag,
        beta=beta,
        normalize_madgwick_step=normalize_madgwick_step,
        forward=False)[0]
    return madgwick(
        timestamps=ts, acc=acc, gyr=gyr, mag=mag, earth_vector=mag_vect,
        q0=q0_forward,
        gain_acc=gain_acc,
        gain_mag=gain_mag,
        beta=beta,
        normalize_madgwick_step=normalize_madgwick_step,
        forward=True)


def ahrs_madgwick_rust(ts, mag, gyr, acc, mag_vect, gain_acc, gain_mag, beta, normalize_madgwick_step):
    import skipper_madgwick_filter_rs
    config = skipper_madgwick_filter_rs.ConfigRs(
        start=skipper_madgwick_filter_rs.StartingRs(
            start_type="compute",
            duration=60,
            gains=skipper_madgwick_filter_rs.GainsRs(
                acc=skipper_madgwick_filter_rs.GainTypeRs(
                    gain_type="dynamic",
                    value=-5*1e-4,
                ),
                mag=skipper_madgwick_filter_rs.GainTypeRs(
                    gain_type="dynamic",
                    value=-5*1e-4,
                ),
                gyr=skipper_madgwick_filter_rs.GainTypeRs(
                    gain_type="constant",
                    value=0.5,
                ),
            ),
            quaternion=None,
        ),
        gains=skipper_madgwick_filter_rs.GainsRs(
            acc=skipper_madgwick_filter_rs.GainTypeRs(
                gain_type="constant",
                value=-numpy.max(gain_acc),
            ),
            mag=skipper_madgwick_filter_rs.GainTypeRs(
                gain_type="constant",
                value=-numpy.max(gain_mag),
            ),
            gyr=skipper_madgwick_filter_rs.GainTypeRs(
                gain_type="constant",
                value=0.5,
            ),
        ),
    )
    rot_mat = skipper_madgwick_filter_rs.compute_rotations(
        timestamps=numpy.ascontiguousarray(ts),
        acc=numpy.ascontiguousarray(acc),
        gyr=numpy.ascontiguousarray(gyr),
        mag=numpy.ascontiguousarray(mag, dtype=numpy.float64),
        earth_vector=mag_vect.tolist(),
        config=config,
    )
    quaternions = rotations.rot_mat_to_quaternions(rot_mat)
    return quaternions


def export_to_laz(
        path,
        mxlf_ned, mylf_ned, mzlf_ned,
        mxla_ned, myla_ned, mzla_ned,
        mxra_ned, myra_ned, mzra_ned,
        mxrf_ned, myrf_ned, mzrf_ned,
        positions, epsg):

    # plt.scatter(positions[0][0][0], positions[0][0][1], c=mzlf_ned, cmap='rainbow_r')
    # plt.show()

    X = numpy.array([position[0][0] for position in positions]).flatten()
    Y = numpy.array([position[0][1] for position in positions]).flatten()
    Z = numpy.array([position[0][2] for position in positions]).flatten()

    mx = numpy.array([mxlf_ned, mxla_ned, mxra_ned, mxrf_ned]).flatten()
    my = numpy.array([mylf_ned, myla_ned, myra_ned, myrf_ned]).flatten()
    mz = numpy.array([mzlf_ned, mzla_ned, mzra_ned, mzrf_ned]).flatten()
    mn = numpy.linalg.norm([mx, my, mz], axis=0)

    # plt.scatter(X, Y, c=mz, cmap='rainbow_r')
    # plt.show()

    values = {
        "X": X,
        "Y": Y,
        "Z": Z,
        "Mx": mx,
        "My": my,
        "Mz": mz,
        "Mn": mn,
    }
    return point_cloud.create_point_cloud(path, values, epsg)

