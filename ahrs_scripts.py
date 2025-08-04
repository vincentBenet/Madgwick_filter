import numpy as np
import utils


def ahrs_madgwick_rust(ts, mag, gyr, acc, mag_vect, gain_acc, gain_mag):
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
                value=-gain_acc,
            ),
            mag=skipper_madgwick_filter_rs.GainTypeRs(
                gain_type="constant",
                value=-gain_mag,
            ),
            gyr=skipper_madgwick_filter_rs.GainTypeRs(
                gain_type="constant",
                value=0.5,
            ),
        ),
    )
    rot_mat = skipper_madgwick_filter_rs.compute_rotations(
        timestamps=np.ascontiguousarray(ts),
        acc=np.ascontiguousarray(acc),
        gyr=np.ascontiguousarray(gyr),
        mag=np.ascontiguousarray(mag, dtype=np.float64),
        earth_vector=mag_vect.tolist(),
        config=config,
    )
    quaternions = utils.rot_mat_to_quaternions(rot_mat)
    return quaternions


def ahrs_madgwick_python_benet(ts, mag, gyr, acc, mag_vect, gain_acc, gain_mag, acc_weight, mag_weight):
    q0_sample = int(60 / np.median(np.diff(ts)))
    q0_backward = utils.madgwick(
        timestamps=ts[-q0_sample:],
        acc=acc[-q0_sample:],
        gyr=gyr[-q0_sample:],
        mag=mag[-q0_sample:],
        earth_vector=mag_vect,
        q0=None,
        mag_weight=mag_weight,
        acc_weight=acc_weight,
        gain_acc=gain_acc,
        gain_mag=gain_mag,
        forward=True)[-1]
    q0_forward = utils.madgwick(
        timestamps=ts, acc=acc, gyr=gyr, mag=mag, earth_vector=mag_vect,
        q0=q0_backward,
        gain_acc=gain_acc,
        mag_weight=mag_weight,
        acc_weight=acc_weight,
        gain_mag=gain_mag,
        forward=False)[0]
    return utils.madgwick(
        timestamps=ts, acc=acc, gyr=gyr, mag=mag, earth_vector=mag_vect,
        q0=q0_forward,
        gain_acc=gain_acc,
        mag_weight=mag_weight,
        acc_weight=acc_weight,
        gain_mag=gain_mag,
        forward=True)
