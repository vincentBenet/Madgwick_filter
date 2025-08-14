import os

import numpy
from matplotlib import pyplot as plt

import utils

path_folder = r"C:\Users\VincentBenet\Documents\NAS_SKIPPERNDT\Share - SkipperNDT\SKPFR_TST_correctionAssiette_Courcouronnes\2025-08-08T10-04-45_4c62d0-373-calib-imu"

ts_imu, ax, ay, az, vrx, vry, vrz = utils.load_parquet_imu(path_folder)

ax_lp = utils.low_pass(ax, ts_imu, t=1)
ay_lp = utils.low_pass(ay, ts_imu, t=1)
az_lp = utils.low_pass(az, ts_imu, t=1)

jx = numpy.diff(ax_lp) / numpy.diff(ts_imu)
jy = numpy.diff(ay_lp) / numpy.diff(ts_imu)
jz = numpy.diff(az_lp) / numpy.diff(ts_imu)

jn = numpy.linalg.norm([jx, jy, jz], axis=0)

mask = jn < numpy.percentile(jn, 10)
plt.plot(ts_imu[1:], jn)
plt.show()

ax_filtered = ax_lp[1:][mask]
ay_filtered = ay_lp[1:][mask]
az_filtered = az_lp[1:][mask]

m, b = utils.calcul_calibration(
    ax_filtered,
    ay_filtered,
    az_filtered,
    numpy.array([0, 0, 1]))

print(f"{m = }")
print(f"{b = }")

ax_calibrated, ay_calibrated, az_calibrated = utils.apply_calibration((m, b), ax_filtered, ay_filtered, az_filtered)

norm_uncalibrated = numpy.linalg.norm([ax_filtered, ay_filtered, az_filtered], axis=0)
norm_calibrated = numpy.linalg.norm([ax_calibrated, ay_calibrated, az_calibrated], axis=0)

std_uncalibrated = numpy.std(norm_uncalibrated)
std_calibrated = numpy.std(norm_calibrated)

print(f"{std_uncalibrated = }")
print(f"{std_calibrated = }")

print(f"Improvment x {round(std_uncalibrated / std_calibrated, 2)}")

plt.scatter(ts_imu[1:][mask], norm_uncalibrated, label=f"Uncalibrated: {round(std_uncalibrated, 5)}g")
plt.scatter(ts_imu[1:][mask], norm_calibrated, label=f"Calibrated: {round(std_calibrated, 5)}g")
plt.legend()
plt.show()

fig = plt.figure(figsize=(12, 12))
axs = fig.add_subplot(projection='3d')
axs.scatter(ax_filtered, ay_filtered, az_filtered)
axs.scatter(ax_calibrated, ay_calibrated, az_calibrated)
plt.axis("equal")
plt.show()

