"""Production script for diffsky_v0.3
"""
import argparse
import os
from time import time

import numpy as np
from astropy.table import Table
from c3dev.galmocks.data_loaders.load_unit_sims import (
    UNIT_LBOX,
    load_value_added_unit_sim,
)
from c3dev.galmocks.galhalo_models.assign_elliptical_velocities import (
    assign_ellipsoidal_velocities,
)
from c3dev.galmocks.galhalo_models.galsampler_phase_space import (
    add_central_velbias,
    inherit_host_centric_posvel,
)
from c3dev.galmocks.utils import galmatch
from c3dev.galmocks.utils.galprops import compute_lg_ssfr
from diffdesi.load_umachine_utils import get_value_added_um_data, load_umachine_diffsky
from halotools.empirical_models.phase_space_models import NFWPhaseSpace
from halotools.utils import crossmatch, sliding_conditional_percentile
from jax import random as jran

UM_LOGSM_CUT = 9.0
SEED = 43
OUTDRN = "/home/gbeltzmohrmann/CSC3"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("unit_sim_fn", help="path to unit sim subhalo catalog")
    parser.add_argument("um_fn", help="path to UM mock")
    parser.add_argument("um_path", help="path name of UM mock")
    parser.add_argument("um_redshift", type=float, help="UM redshift")
    parser.add_argument("unit_redshift", type=float, help="UNIT redshift")
    parser.add_argument("outname", help="Output fname")
    args = parser.parse_args()

    unit_redshift = args.unit_redshift

    t0 = time()

    ran_key = jran.PRNGKey(SEED)

    # Pick up the mock from disk
    um_mock = load_umachine_diffsky(args.um_fn, args.um_path)

    # Stellar mass cut
    logsm_msk = um_mock["mstar"] > 10**UM_LOGSM_CUT
    um_mock = um_mock[logsm_msk]

    # The GalSampler algorithm needs both host halos and the galaxies occupying them
    um_mock, um_halos = get_value_added_um_data(um_mock)

    # Time
    t1 = time()
    print("{0:.1f} total seconds to load value-added UM".format(t1 - t0))

    # Load Unit sim
    unit = load_value_added_unit_sim(args.unit_sim_fn)

    # Time
    t2 = time()
    print("{0:.1f} seconds to load UNIT".format(t2 - t1))

    # Hosts
    unit = unit[unit["halo_upid"] == -1]

    # Compute Prob(<Vmax | Mvir) for host halos in Unit
    nfw = NFWPhaseSpace(redshift=unit_redshift)
    unit["halo_vvir"] = nfw.virial_velocity(unit["halo_mvir"])
    unit["p_conc"] = sliding_conditional_percentile(
        np.log10(unit["halo_mvir"]), unit["halo_nfw_conc"], 201
    )

    # Compute Prob(<Vmax | Mvir) for host halos in SMDPL
    um_halos["p_vmax"] = sliding_conditional_percentile(
        um_halos["log_Mvir_host"], um_halos["Vmax_host"], 201
    )

    # Run GalSampler to compute index gymnastics for the halo--halo correspondence
    source_galaxies_host_halo_id = um_mock["Host_id"]
    source_halo_ids = um_halos["halo_id"]
    target_halo_ids = unit["halo_id"]
    source_halo_props = (um_halos["log_Mvir_host"], um_halos["p_vmax"])
    target_halo_props = (np.log10(unit["halo_mvir"]), unit["p_conc"])
    d = (
        source_galaxies_host_halo_id,
        source_halo_ids,
        target_halo_ids,
        source_halo_props,
        target_halo_props,
    )
    t5 = time()
    galsampler_res = galmatch.compute_source_galaxy_selection_indices(*d)
    t6 = time()
    print("{0:.1f} seconds to galsample".format(t6 - t5))

    # output mock inherits properties from UNIT host halos
    keys_to_inherit_from_unit = (
        "halo_x",
        "halo_y",
        "halo_z",
        "halo_vx",
        "halo_vy",
        "halo_vz",
        "halo_mvir",
        "halo_vvir",
        "halo_rvir",
        "halo_upid",
        "halo_id",
        "halo_nfw_conc",
        "halo_a_x",
        "halo_a_y",
        "halo_a_z",
        "halo_b_to_a",
        "halo_c_to_a",
    )
    n_output_mock = galsampler_res.target_gals_target_halo_ids.size
    idxA, idxB = crossmatch(galsampler_res.target_gals_target_halo_ids, unit["halo_id"])
    output_mock = Table()
    for key in keys_to_inherit_from_unit:
        output_mock["unit_" + key] = np.zeros(n_output_mock)
        output_mock["unit_" + key][idxA] = unit[key][idxB]
    t7 = time()
    print("{0:.1f} seconds to inherit from unit with crossmatch".format(t7 - t6))

    # Inherit from Umachine Diffsky
    keys_to_inherit_from_diffsky = [
        "is_cen",
        "sub_pos",
        "sub_vel",
        "Host_pos",
        "Host_vel",
        "log_Mvir_host",
        "Vmax_host",
        "mstar",
        "app_mag_g",
        "app_mag_r",
        "app_mag_z",
        "app_mag_wise1",
        "abs_mag_g",
        "abs_mag_r",
        "abs_mag_z",
        "abs_mag_wise1",
        "isLRG",
    ]
    for key in keys_to_inherit_from_diffsky:
        output_mock["diffsky_" + key] = um_mock[key][
            galsampler_res.target_gals_selection_indx
        ]

    # Inherit halo IDs for possible future sanity checks on the bookkeeping
    _key = "galsampler_target_halo_ids"
    output_mock[_key] = galsampler_res.target_gals_target_halo_ids

    _key = "galsampler_source_halo_ids"
    output_mock[_key] = galsampler_res.target_gals_source_halo_ids
    t8 = time()
    print("{0:.1f} seconds to inherit from UM".format(t8 - t7))

    # Assign default phase space model
    logmh_host_target = np.log10(output_mock["unit_halo_mvir"])
    pos_host_target = np.vstack(
        (
            output_mock["unit_halo_x"],
            output_mock["unit_halo_y"],
            output_mock["unit_halo_z"],
        )
    ).T
    vel_host_target = np.vstack(
        (
            output_mock["unit_halo_vx"],
            output_mock["unit_halo_vy"],
            output_mock["unit_halo_vz"],
        )
    ).T

    is_cen_target = output_mock["diffsky_is_cen"].astype("bool")
    is_sat_target = ~is_cen_target

    is_cen_source = um_mock["is_cen"].astype("bool")
    is_sat_source = ~is_cen_source
    logmh_host_source = um_mock["log_Mvir_host"]
    pos_host_source = um_mock["Host_pos"]
    vel_host_source = um_mock["Host_vel"]
    pos_source = um_mock["sub_pos"]
    vel_source = um_mock["sub_vel"]
    ran_key, posvel_key = jran.split(ran_key, 2)
    pos_target, vel_target = inherit_host_centric_posvel(
        posvel_key,
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
    )
    output_mock["pos"] = np.mod(pos_target, UNIT_LBOX)
    output_mock["vel"] = vel_target

    is_cen_target = output_mock["diffsky_is_cen"]
    vel_source = output_mock["diffsky_sub_vel"]
    vel_host_source = output_mock["diffsky_Host_vel"]
    vel_host_target = output_mock["vel"]
    output_mock["vel"] = add_central_velbias(
        is_cen_target, vel_source, vel_host_source, vel_host_target
    )
    t9 = time()
    print("{0:.1f} seconds to assign phase space".format(t9 - t8))

    ran_key, ellipsoidal_key = jran.split(ran_key, 2)
    t10 = time()
    output_mock["tng_is_central"] = output_mock["diffsky_is_cen"]
    assign_ellipsoidal_velocities(
        ellipsoidal_key, output_mock, unit_redshift, UNIT_LBOX
    )
    t11 = time()
    print("{0:.1f} seconds to assign ellipsoidal velocities".format(t11 - t10))

    print("\n")
    print(output_mock.keys())
    outname = os.path.join(OUTDRN, args.outname)
    output_mock.write(outname, path="data", overwrite=True)
    print("\n")
