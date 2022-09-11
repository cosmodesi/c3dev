"""
"""
import numpy as np
from jax import random as jran
from .ellipsoidal_nfw_phase_space import mc_ellipsoidal_nfw
from .vector_utilities import normalized_vectors


def assign_ellipsoidal_velocities(ran_key, mock, redshift, Lbox):

    cenmsk = mock["tng_is_central"] == 1
    n_sats = (~cenmsk).sum()

    conc = mock["unit_halo_nfw_conc"][~cenmsk]

    ax = mock["unit_halo_a_x"][~cenmsk]
    ay = mock["unit_halo_a_y"][~cenmsk]
    az = mock["unit_halo_a_z"][~cenmsk]
    major_axes = np.vstack((ax, ay, az)).T

    rvir = mock["unit_halo_rvir"][~cenmsk]
    b_to_a = mock["unit_halo_b_to_a"][~cenmsk]
    c_to_a = mock["unit_halo_c_to_a"][~cenmsk]
    sigma = mock["unit_halo_vvir"][~cenmsk]

    ran_key, orientation_key = jran.split(ran_key)
    random_orientations = normalized_vectors(
        np.array(jran.uniform(orientation_key, shape=(n_sats, 3)))
    )

    host_x = mock["unit_halo_x"]
    host_y = mock["unit_halo_y"]
    host_z = mock["unit_halo_z"]

    host_vx = mock["unit_halo_vx"]
    host_vy = mock["unit_halo_vy"]
    host_vz = mock["unit_halo_vz"]

    # LSS correlated intra-halo NFW positions
    nfw_host_centric_pos, nfw_host_centric_vel = mc_ellipsoidal_nfw(
        ran_key, rvir, conc, sigma, major_axes, b_to_a, c_to_a
    )
    host_pos = np.vstack((host_x, host_y, host_z)).T
    host_pos[~cenmsk] = host_pos[~cenmsk] + nfw_host_centric_pos
    mock["pos_model1"] = np.mod(host_pos, Lbox)

    host_vel = np.vstack((host_vx, host_vy, host_vz)).T
    host_vel[~cenmsk] = host_vel[~cenmsk] + nfw_host_centric_vel
    mock["vel_model1"] = host_vel

    # LSS uncorrelated intra-halo NFW positions
    nfw_host_centric_pos, nfw_host_centric_vel = mc_ellipsoidal_nfw(
        ran_key, rvir, conc, sigma, random_orientations, b_to_a, c_to_a
    )
    host_pos = np.vstack((host_x, host_y, host_z)).T
    host_pos[~cenmsk] = host_pos[~cenmsk] + nfw_host_centric_pos
    mock["pos_model2"] = np.mod(host_pos, Lbox)

    host_vel = np.vstack((host_vx, host_vy, host_vz)).T
    host_vel[~cenmsk] = host_vel[~cenmsk] + nfw_host_centric_vel
    mock["vel_model2"] = host_vel

    # spherical NFW intra-halo positions
    b_to_a = np.ones(n_sats)
    c_to_a = np.ones(n_sats)
    nfw_host_centric_pos, nfw_host_centric_vel = mc_ellipsoidal_nfw(
        ran_key, rvir, conc, sigma, random_orientations, b_to_a, c_to_a
    )
    host_pos = np.vstack((host_x, host_y, host_z)).T
    host_pos[~cenmsk] = host_pos[~cenmsk] + nfw_host_centric_pos
    mock["pos_model3"] = np.mod(host_pos, Lbox)

    host_vel = np.vstack((host_vx, host_vy, host_vz)).T
    host_vel[~cenmsk] = host_vel[~cenmsk] + nfw_host_centric_vel
    mock["vel_model3"] = host_vel

    return mock
