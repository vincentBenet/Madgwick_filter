import os
import numpy
import numpy as np
from matplotlib import pyplot as plt

import rotations
import utils


def main(
    path_folder, path_laz_LF_ENU, path_calibration, ts_mag_to_imu, ahrs_func,
    duration_filter_mag_axis, duration_filter_mag_merged, duration_filter_gyr, n_filter_acc,
    duration_filter_quaternions_outputs, gain_acc, gain_mag, plot
):
    try:  # Load Orion result point cloud (need python of QGIS 3.40 to execute)
        assert path_laz_LF_ENU is not None
        from pyqgis_tools import point_cloud
        values_lf_enu = point_cloud.load_points(["bx_nT", "by_nT", "bz_nT", "GpsTime"], path=path_laz_LF_ENU)
        _, values_lf_enu["bx_nT"] = zip(*sorted(zip(values_lf_enu["GpsTime"], values_lf_enu["bx_nT"])))
        _, values_lf_enu["by_nT"] = zip(*sorted(zip(values_lf_enu["GpsTime"], values_lf_enu["by_nT"])))
        values_lf_enu["GpsTime"], values_lf_enu["bz_nT"] = zip(*sorted(zip(values_lf_enu["GpsTime"], values_lf_enu["bz_nT"])))

        values_lf_enu["bx_nT"] = utils.low_pass(values_lf_enu["bx_nT"], ts=values_lf_enu["GpsTime"], t=duration_filter_mag_axis)
        values_lf_enu["by_nT"] = utils.low_pass(values_lf_enu["by_nT"], ts=values_lf_enu["GpsTime"], t=duration_filter_mag_axis)
        values_lf_enu["bz_nT"] = utils.low_pass(values_lf_enu["bz_nT"], ts=values_lf_enu["GpsTime"], t=duration_filter_mag_axis)

        std_orion = numpy.std(values_lf_enu["bx_nT"]) + numpy.std(values_lf_enu["by_nT"]) + numpy.std(
            values_lf_enu["bz_nT"])
        env_x_orion = utils.get_enveloppe(values_lf_enu["GpsTime"], values_lf_enu["bx_nT"])
        env_y_orion = utils.get_enveloppe(values_lf_enu["GpsTime"], values_lf_enu["by_nT"])
        env_z_orion = utils.get_enveloppe(values_lf_enu["GpsTime"], values_lf_enu["bz_nT"])
        env_orion = numpy.mean(env_x_orion + env_y_orion + env_z_orion)
        cloud = True
    except Exception as e:
        print(e)
        values_lf_enu = None
        env_x_orion = numpy.zeros(3)
        env_y_orion = numpy.zeros(3)
        env_z_orion = numpy.zeros(3)
        env_orion = 0
        std_orion = 0
        cloud = False

    (  # Load RAW data
        ts_gnss, gnss_x, gnss_y, gnss_z, gnss_lon, gnss_lat,
        ts_imu, ax, ay, az, vrx, vry, vrz,
        ts_mag, mxlf, mylf, mzlf, mxla, myla, mzla, mxra, myra, mzra, mxrf, myrf, mzrf
    ) = utils.load_parquet_data(path_folder, path_calibration)

    ts, quaternions, mx_interp, my_interp, mz_interp, ax_interp, ay_interp, az_interp, mag_vect_raw = utils.ahrs(
        gnss_z, gnss_lon, gnss_lat,
        ts_imu, ax, ay, az, vrx, vry, vrz,
        ts_mag, mxlf, mylf, mzlf, mxla, myla, mzla, mxra, myra, mzra, mxrf, myrf, mzrf,
        duration_filter_mag_axis, duration_filter_mag_merged, duration_filter_gyr, n_filter_acc,
        duration_filter_quaternions_outputs, gain_acc, gain_mag, path_calibration, ts_mag_to_imu, ahrs_func,
    )

    mx_ned, my_ned, mz_ned = rotations.rotate_data(mx_interp, my_interp, mz_interp, quaternions)

    std_upgrade = numpy.std(mx_ned) + numpy.std(my_ned) + numpy.std(mz_ned)

    env_x = utils.get_enveloppe(ts, mx_ned)
    env_y = utils.get_enveloppe(ts, my_ned)
    env_z = utils.get_enveloppe(ts, mz_ned)

    ax_ned, ay_ned, az_ned = rotations.rotate_data(ax_interp, ay_interp, az_interp, quaternions)

    yaw, pitch, roll = rotations.quaternion_to_euler(quaternions)
    duration_yaw_filter = 2
    yaw_gnss = rotations.gnss_to_yaw(gnss_x, gnss_y, ts_gnss, duration_yaw_filter)
    yaw_imu = utils.low_pass(yaw, ts=ts, t=duration_yaw_filter)

    # plt.scatter(ts_gnss[1:], yaw_gnss, label="yaw_gnss")
    # plt.plot(ts, yaw_imu, label="yaw")
    # plt.plot(ts, pitch, label="pitch")
    # plt.plot(ts, roll, label="roll")
    # plt.legend()
    # plt.show()


    env_upgrade = numpy.mean(env_x + env_y + env_z)

    avg_acc_n = numpy.median(ax_ned)
    avg_acc_e = numpy.median(ay_ned)
    avg_acc_d = numpy.median(az_ned)
    error_acc_n = numpy.abs(avg_acc_n)
    error_acc_e = numpy.abs(avg_acc_e)
    error_acc_d = numpy.abs(avg_acc_d-1)
    sum_acc_error = error_acc_n + error_acc_e + error_acc_d

    print(f"{avg_acc_n = }")
    print(f"{avg_acc_e = }")
    print(f"{avg_acc_d = }")
    print(f"{error_acc_n = }")
    print(f"{error_acc_e = }")
    print(f"{error_acc_d = }")
    print(f"{sum_acc_error = }")

    avg_mag_n = numpy.median(mx_ned)
    avg_mag_e = numpy.median(my_ned)
    avg_mag_d = numpy.median(mz_ned)

    error_mag_n = abs(avg_mag_n - mag_vect_raw[0])
    error_mag_e = abs(avg_mag_e - mag_vect_raw[1])
    error_mag_d = abs(avg_mag_d - mag_vect_raw[2])
    sum_mag_error = error_mag_n + error_mag_e + error_mag_d

    print(f"{sum_mag_error = } nT")

    print(f"STD: {round(std_orion, 1)} - {round(std_upgrade, 1)} (-{round(100 * (1 - std_upgrade / std_orion), 1)}%)\nENV: {round(env_orion, 1)} - {round(env_upgrade, 1)} (-{round(100 * (1 - env_upgrade / env_orion), 1)}%)")

    if plot:

        # plt.plot(ts, ax_ned, label="ax")
        # plt.plot(ts, ay_ned, label="ay")
        # plt.plot(ts, az_ned, label="az")
        # plt.grid(True)
        # plt.legend()
        # plt.show()

        # plt.plot(ts_gnss[1:], yaw_gnss)
        # plt.plot(ts, yaw_imu)
        # plt.show()

        fig, axs = plt.subplots(3)

        if cloud:
            axs[0].scatter(
                values_lf_enu["GpsTime"] - min(values_lf_enu["GpsTime"]),
                values_lf_enu["bx_nT"] - numpy.mean(values_lf_enu["bx_nT"]),
                label=f"Bx Ref [{round(np.max(values_lf_enu["bx_nT"]) - np.min(values_lf_enu["bx_nT"]), 1)}, {round(np.std(values_lf_enu["bx_nT"]), 1)}]",
                color="red")
            axs[1].scatter(
                values_lf_enu["GpsTime"] - min(values_lf_enu["GpsTime"]),
                values_lf_enu["by_nT"] - numpy.mean(values_lf_enu["by_nT"]),
                label=f"By Ref [{round(np.max(values_lf_enu["by_nT"]) - np.min(values_lf_enu["by_nT"]), 1)}, {round(np.std(values_lf_enu["by_nT"]), 1)}]",
                color="red")
            axs[2].scatter(
                values_lf_enu["GpsTime"] - min(values_lf_enu["GpsTime"]),
                values_lf_enu["bz_nT"] - numpy.mean(values_lf_enu["bz_nT"]),
                label=f"Bz Ref [{round(np.max(values_lf_enu["bz_nT"]) - np.min(values_lf_enu["bz_nT"]), 1)}, {round(np.std(values_lf_enu["bz_nT"]), 1)}]",
                color="red")
            axs[0].plot(
                values_lf_enu["GpsTime"] - values_lf_enu["GpsTime"][0],
                env_x_orion,
                label=f"Bx env Ref {round(numpy.mean(env_x_orion), 1)} nT",
                color="orange",
                linestyle="--")
            axs[1].plot(
                values_lf_enu["GpsTime"] - values_lf_enu["GpsTime"][0],
                env_y_orion,
                label=f"By env Ref {round(numpy.mean(env_y_orion), 1)} nT",
                color="orange",
                linestyle="--")
            axs[2].plot(
                values_lf_enu["GpsTime"] - values_lf_enu["GpsTime"][0],
                env_z_orion,
                label=f"Bz env Ref {round(numpy.mean(env_z_orion), 1)} nT",
                color="orange",
                linestyle="--")

        axs[0].plot(ts - ts[0], env_x, label=f"Bx env {round(numpy.mean(env_x), 1)} nT", color="blue", linestyle="--")
        axs[1].plot(ts - ts[0], env_y, label=f"By env {round(numpy.mean(env_y), 1)} nT", color="blue", linestyle="--")
        axs[2].plot(ts - ts[0], env_z, label=f"Bz env {round(numpy.mean(env_z), 1)} nT", color="blue", linestyle="--")
        axs[0].plot(ts - ts[0], mx_ned - numpy.mean(mx_ned), label=f"Bx [{round(numpy.max(mx_ned) - numpy.min(mx_ned), 1)}, {round(numpy.std(mx_ned), 1)}]", color="green")
        axs[1].plot(ts - ts[0], my_ned - numpy.mean(my_ned), label=f"By [{round(numpy.max(my_ned) - numpy.min(my_ned), 1)}, {round(numpy.std(my_ned), 1)}]", color="green")
        axs[2].plot(ts - ts[0], mz_ned - numpy.mean(mz_ned), label=f"Bz [{round(numpy.max(mz_ned) - numpy.min(mz_ned), 1)}, {round(numpy.std(mz_ned), 1)}]", color="green")

        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)
        fig.suptitle(
            f"{path_folder.split(os.sep)[-1].split("/")[-1]}\nSTD: {round(std_orion, 1)} - {round(std_upgrade, 1)} (-{round(100 * (1 - std_upgrade / std_orion), 1)}%)\nENV: {round(env_orion, 1)} - {round(env_upgrade, 1)} (-{round(100 * (1 - env_upgrade / env_orion), 1)}%)")
        plt.show()

        rotations.mag_position(ts_gnss, gnss_x, gnss_y, gnss_z, ts, quaternions)

    return env_upgrade, env_orion, std_upgrade, std_orion


if __name__ == "__main__":
    plot = True
    folder = r"C:\Users\VincentBenet\Documents\NAS_SKIPPERNDT\Share - SkipperNDT\SKPFR_TST_correctionAssiette_Courcouronnes"
    configs = {
        # "Susville-champ-2": {"calib": os.path.join(folder, "2025-07-30T12-10-49_4c62d0-367-Susville-calib"),
        #                      "laz": os.path.join(folder, r"2025-07-30T11-58-47_4c62d0-366-Susville-champ-2\gis\point_cloud_mag_5fa74312_ENU.laz")},
        # "Susville-champ-1": {"calib": os.path.join(folder, "2025-07-30T12-10-49_4c62d0-367-Susville-calib"),
        #                      "laz": os.path.join(folder, r"2025-07-29T15-11-24_4c62d0-365-Susville-champ-1\gis\point_cloud_mag_59080c1c_ENU.laz")},
        # "Courcourronnes-pente-marche": {"calib": os.path.join(folder, "2025-07-03T10-53-30_4c841c-586-backpack-calib"),
        #                                 "laz": os.path.join(folder, r"2025-07-03T10-57-24_4c841c-587-backpack-pente-marche\gis\point_cloud_mag_141eb8c6_ENU.laz")},
        # "Courcourronnes-pente-course": {"calib": os.path.join(folder, "2025-07-03T10-53-30_4c841c-586-backpack-calib"),
        #                                 "laz": os.path.join(folder, r"2025-07-03T11-13-39_4c841c-588-backpack-pente-course\gis\point_cloud_mag_a402d77c_ENU.laz")},
        "Courcourronnes-plat-marche": {"calib": os.path.join(folder, "2025-07-03T10-53-30_4c841c-586-backpack-calib"),
                                       "laz": os.path.join(folder, r"2025-07-03T11-22-01_4c841c-589-backpack-plat-marche\gis\point_cloud_mag_6743ee20_ENU.laz")},
        # "Courcourronnes-plat-course": {"calib": os.path.join(folder, "2025-07-03T10-53-30_4c841c-586-backpack-calib"),
        #                                "laz": os.path.join(folder, r"2025-07-03T11-32-46_4c841c-590-backpack-plat-course\gis\point_cloud_mag_809cc083_ENU.laz")},
        # "vaulnavey-foret": {"calib": os.path.join(folder, "2025-08-01T09-30-20_4c62d0-369-vaulnavey-calib"),
        #                                "laz": os.path.join(folder, r"2025-08-01T09-30-20_4c62d0-369-vaulnavey-calib\gis\point_cloud_mag_1fbd1b88_ENU.laz")},
        # "vaulnavey-calib": {"calib": os.path.join(folder, "2025-08-01T09-30-20_4c62d0-369-vaulnavey-calib"),
        #                                "laz": os.path.join(folder, r"2025-08-01T09-08-18_4c62d0-368-vaulnavey-foret\gis\point_cloud_mag_8a93dd05_ENU.laz")},
        # "Susville-calib": {"calib": os.path.join(folder, "2025-07-30T12-10-49_4c62d0-367-Susville-calib"),
        #                    "laz": os.path.join(folder, r"2025-07-30T12-10-49_4c62d0-367-Susville-calib\gis\point_cloud_mag_7bee0a55_ENU.laz")},
        # "Longnes-calib": {"calib": os.path.join(folder, "2024-12-18T15-23-32_4c841c-384-Longnes-calib"),
        #                   "laz": os.path.join(folder, r"2024-12-18T15-23-32_4c841c-384-Longnes-calib\gis\point_cloud_mag_42acc7a9_ENU.laz")},
        # "Francisville-calib": {"calib": os.path.join(folder, "2025-03-13T17-36-01_290dc9-80-Francisville-calib"),
        #                          "laz": os.path.join(folder, r"2025-03-13T17-36-01_290dc9-80-Francisville-calib\gis\point_cloud_mag_f9993255_ENU.laz")},
        # "Courcourronnes-calib": {"calib": os.path.join(folder, "2025-07-03T10-53-30_4c841c-586-backpack-calib"),
        #                          "laz": os.path.join(folder, r"2025-07-03T10-53-30_4c841c-586-backpack-calib\gis\point_cloud_mag_4431dee4_ENU.laz")},
        # "Hope-calib": {"calib": os.path.join(folder, "2025-02-25T21-32-12_290dc9-34-Hope-calib"),
        #                          "laz": os.path.join(folder, r"2025-02-25T21-32-12_290dc9-34-Hope-calib\gis\point_cloud_mag_c2f1af84_ENU.laz")},
        # "SmartPipe-calib": {"calib": os.path.join(folder, "2025-02-21T00-00-00_290dc9-31-SmartPipe-calib"),
        #                          "laz": os.path.join(folder, r"2025-02-21T00-00-00_290dc9-31-SmartPipe-calib\gis\point_cloud_mag_8aca4299_ENU.laz")},
        # "SmartPipe-z1": {"calib": os.path.join(folder, "2025-02-21T00-00-00_290dc9-31-SmartPipe-calib"),
        #                          "laz": os.path.join(folder, r"2025-02-20T13-29-08_290dc9-19-SmartPipe-z1\gis\point_cloud_mag_66db9249_ENU.laz")},
        # "SmartPipe-z2": {"calib": os.path.join(folder, "2025-02-21T00-00-00_290dc9-31-SmartPipe-calib"),
        #                          "laz": os.path.join(folder, r"2025-02-20T13-38-29_290dc9-20-SmartPipe-z2\gis\point_cloud_mag_d4d11929_ENU.laz")},
        # "SmartPipe-z3": {"calib": os.path.join(folder, "2025-02-21T00-00-00_290dc9-31-SmartPipe-calib"),
        #                          "laz": os.path.join(folder, r"2025-02-20T14-16-01_290dc9-21-SmartPipe-z3\gis\point_cloud_mag_b619d263_ENU.laz")},
        # "SmartPipe-z4": {"calib": os.path.join(folder, "2025-02-21T00-00-00_290dc9-31-SmartPipe-calib"),
        #                          "laz": os.path.join(folder, r"2025-02-20T14-43-54_290dc9-22-SmartPipe-z4\gis\point_cloud_mag_5a720537_ENU.laz")},
        # "SmartPipe-z5": {"calib": os.path.join(folder, "2025-02-21T00-00-00_290dc9-31-SmartPipe-calib"),
        #                          "laz": os.path.join(folder, r"2025-02-20T15-05-28_290dc9-23-SmartPipe-z5\gis\point_cloud_mag_ee8b2f68_ENU.laz")},
        # "SmartPipe-z6": {"calib": os.path.join(folder, "2025-02-21T00-00-00_290dc9-31-SmartPipe-calib"),
        #                          "laz": os.path.join(folder, r"2025-02-20T15-41-34_290dc9-24-SmartPipe-z6\gis\point_cloud_mag_066c2d08_ENU.laz")},
        # "SmartPipe-z7": {"calib": os.path.join(folder, "2025-02-21T00-00-00_290dc9-31-SmartPipe-calib"),
        #                          "laz": os.path.join(folder, r"2025-02-20T15-59-31_290dc9-25-SmartPipe-z7\gis\point_cloud_mag_a5eed459_ENU.laz")},
        # "SmartPipe-z8": {"calib": os.path.join(folder, "2025-02-21T00-00-00_290dc9-31-SmartPipe-calib"),
        #                          "laz": os.path.join(folder, r"2025-02-20T16-18-02_290dc9-26-SmartPipe-z8\gis\point_cloud_mag_3d9a79a9_ENU.laz")},
        # "SmartPipe-z9": {"calib": os.path.join(folder, "2025-02-21T00-00-00_290dc9-31-SmartPipe-calib"),
        #                          "laz": os.path.join(folder, r"2025-02-20T16-38-43_290dc9-27-SmartPipe-z9\gis\point_cloud_mag_efa5bd5e_ENU.laz")},
        # "Hope-z1": {"calib": os.path.join(folder, "2025-02-25T21-32-12_290dc9-34-Hope-calib"),
        #                          "laz": os.path.join(folder, r"2025-02-25T21-01-27_290dc9-33-Hope-z1\gis\point_cloud_mag_ec89e0a2_ENU.laz")},
        # "Hope-z2": {"calib": os.path.join(folder, "2025-02-25T21-32-12_290dc9-34-Hope-calib"),
        #                          "laz": os.path.join(folder, r"2025-02-25T18-36-45_290dc9-32-Hope-z2\gis\point_cloud_mag_5cfb3384_ENU.laz")},
        # "Francisville-z1": {"calib": os.path.join(folder, "2025-03-13T17-36-01_290dc9-80-Francisville-calib"),
        #                          "laz": os.path.join(folder, r"2025-03-13T14-20-15_290dc9-75-Francisville-z1\gis\point_cloud_mag_035640f9_ENU.laz")},
        # "Francisville-z2": {"calib": os.path.join(folder, "2025-03-13T17-36-01_290dc9-80-Francisville-calib"),
        #                          "laz": os.path.join(folder, r"2025-03-13T15-27-09_290dc9-77-Francisville-z2\gis\point_cloud_mag_3ec0a434_ENU.laz")},
        # "Francisville-z3": {"calib": os.path.join(folder, "2025-03-13T17-36-01_290dc9-80-Francisville-calib"),
        #                          "laz": os.path.join(folder, r"2025-03-13T16-01-00_290dc9-78-Francisville-z3\gis\point_cloud_mag_11fe3e87_ENU.laz")},
        # "Francisville-z4": {"calib": os.path.join(folder, "2025-03-13T17-36-01_290dc9-80-Francisville-calib"),
        #                          "laz": os.path.join(folder, r"2025-03-13T16-23-25_290dc9-79-Francisville-z4\gis\point_cloud_mag_6ad2cfd6_ENU.laz")},
        # "Longnes-foret-pipe": {"calib": os.path.join(folder, "2024-12-18T15-23-32_4c841c-384-Longnes-calib"),
        #                        "laz": os.path.join(folder, r"2025-04-23T10-35-22_4c841c-558-Longnes-foret\gis\point_cloud_mag_7a2eefa4_ENU.laz")},
        # "Longnes-champ-pipe": {"calib": os.path.join(folder, "2024-12-18T15-23-32_4c841c-384-Longnes-calib"),
        #                        "laz": os.path.join(folder, r"2025-04-23T12-01-33_4c841c-560-Longnes-champ\gis\point_cloud_mag_849247dd_ENU.laz")},
    }
    params = {
        "duration_filter_mag_axis": 0.01,
        "duration_filter_mag_merged": 0.05,
        "duration_filter_gyr": 0.05,
        "n_filter_acc": 3,
        "duration_filter_quaternions_outputs": 0.01,
        "gain_acc": 1 * 1e-2,
        "gain_mag": 2 * 1e-1,
    }
    args = {
        "ts_mag_to_imu": True,
        "ahrs_func": utils.ahrs_madgwick_python_benet,
    }

    for key, config in configs.items():
        config["env_upgrade"], config["env_orion"], config["std_upgrade"], config["std_orion"] = main(
            plot=plot,
            path_folder=os.path.dirname(os.path.dirname(config["laz"])),
            path_calibration=config["calib"],
            path_laz_LF_ENU=config["laz"],
            **params,
            **args,
        )

    i = 0
    std = 0
    env = 0
    print("\n" * 5)
    print(f"{params = }")
    print(f"{args = }")
    print("\n")
    for key, config in configs.items():
        ratio_std = 1 - config["std_upgrade"] / config["std_orion"]
        ratio_env = 1 - config["env_upgrade"] / config["env_orion"]
        print(key)
        print(f"\tSTD: {round(config["std_orion"], 1)} -> {round(config["std_upgrade"], 1)} ({-round(ratio_std*100, 1)}%)")
        print(f"\tENV: {round(config["env_orion"], 1)} -> {round(config["env_upgrade"], 1)} ({-round(ratio_env*100, 1)}%)")
        std += config["std_upgrade"]
        env += config["env_upgrade"]
        i += 1
    print("\n")
    print(f"STD AVG: {std/i}nT")
    print(f"ENV AVG: {env/i}nT")
