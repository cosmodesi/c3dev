"""
"""
import numpy as np
from jax import random as jran
from ..utils.galmatch import calculate_indx_correspondence


def inherit_host_centric_posvel(
    ran_key,
    is_sat_source,
    is_sat_target,
    logmh_host_source,
    logmh_host_target,
    pos_host_source,
    pos_host_target,
    vel_host_source,
    vel_host_target,
    pos_source,
    vel_source,
    dlogmh=0.25,
):
    pos_sats, vel_sats = _inherit_host_centric_posvel_satellites(
        ran_key,
        logmh_host_source[is_sat_source],
        logmh_host_target[is_sat_target],
        pos_host_source[is_sat_source],
        pos_host_target[is_sat_target],
        vel_host_source[is_sat_source],
        vel_host_target[is_sat_target],
        pos_source[is_sat_source],
        vel_source[is_sat_source],
        dlogmh,
    )
    pos_target = np.copy(pos_host_target)
    vel_target = np.copy(vel_host_target)

    pos_target[is_sat_target] = pos_sats
    vel_target[is_sat_target] = vel_sats
    return pos_target, vel_target


def _inherit_host_centric_posvel_satellites(
    ran_key,
    logmh_host_source,
    logmh_host_target,
    pos_host_source,
    pos_host_target,
    vel_host_source,
    vel_host_target,
    pos_source,
    vel_source,
    dlogmh,
):
    delta_pos_source = pos_source - pos_host_source
    delta_vel_source = vel_source - vel_host_source

    uran = np.array(
        (jran.uniform(ran_key, shape=logmh_host_target.shape) - 0.5) * dlogmh
    )

    dd_match, indx_match = calculate_indx_correspondence(
        logmh_host_source, logmh_host_target + uran
    )
    delta_pos_target = delta_pos_source[indx_match]
    delta_vel_target = delta_vel_source[indx_match]

    pos_target = pos_host_target + delta_pos_target
    vel_target = vel_host_target + delta_vel_target
    return pos_target, vel_target
