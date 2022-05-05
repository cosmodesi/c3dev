"""Production script for gumbo_v0.1
"""
import os
import numpy as np
import argparse
from time import time
from astropy.table import Table
from jax import random as jran
from c3dev.galmocks.data_loaders.load_tng_data import load_tng_subhalos, TNG_LBOX
from c3dev.galmocks.data_loaders.load_tng_data import load_tng_host_halos
from c3dev.galmocks.data_loaders.load_tng_data import get_value_added_tng_data
from c3dev.galmocks.data_loaders.load_unit_sims import load_value_added_unit_sim
from c3dev.galmocks.data_loaders.load_unit_sims import UNIT_LBOX
from c3dev.galmocks.utils import galmatch, abunmatch
from c3dev.galmocks.galhalo_models.galsampler_phase_space import (
    inherit_host_centric_posvel,
)
from c3dev.galmocks.galhalo_models.galsampler_phase_space import add_central_velbias
from halotools.utils import crossmatch, sliding_conditional_percentile
from c3dev.galmocks.utils.galprops import compute_lg_ssfr

TNG_LOGSM_CUT = 9.0
SEED = 43
OUTDRN = "/lcrc/project/halotools/C3EMC/gumbo"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("unit_sim_fn", help="path to unit sim subhalo catalog")
    parser.add_argument(
        "tng_drn", help="directory to tng data read with illustris_python"
    )
    parser.add_argument("tng_snapnum", type=int, help="TNG snapshot number")
    parser.add_argument("outname", help="Output fname")
    args = parser.parse_args()

    ran_key = jran.PRNGKey(SEED)

    _tng_host_halos = load_tng_host_halos(args.tng_drn, args.tng_snapnum)
    _tng_subhalos = load_tng_subhalos(args.tng_drn, args.tng_snapnum)
    t0 = time()
    _tng, tng_halos = get_value_added_tng_data(_tng_subhalos, _tng_host_halos)

    logsm_msk = _tng["mstar"] > 10**TNG_LOGSM_CUT
    tng = _tng[logsm_msk]
    t1 = time()
    print("{0:.1f} seconds to load TNG".format(t1 - t0))
    unit = load_value_added_unit_sim(args.unit_sim_fn)
    t2 = time()
    print("{0:.1f} seconds to load UNIT".format(t2 - t1))

    unit = unit[unit["halo_upid"] == -1]

    tng_halos["logmh_unit"] = tng_halos["logmh"]

    tng_halos["p_vmax"] = sliding_conditional_percentile(
        tng_halos["logmh_unit"], tng_halos["central_subhalo_vmax"], 201
    )
    unit["p_conc"] = sliding_conditional_percentile(
        np.log10(unit["halo_mvir"]), unit["halo_nfw_conc"], 201
    )

    source_galaxies_host_halo_id = tng["host_halo_index"]
    source_halo_ids = tng_halos["halo_id"]
    target_halo_ids = unit["halo_id"]
    source_halo_props = (tng_halos["logmh_unit"], tng_halos["p_vmax"])
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

    # Inherit from UNIT
    keys_to_inherit_from_unit = (
        "halo_x",
        "halo_y",
        "halo_z",
        "halo_vx",
        "halo_vy",
        "halo_vz",
        "halo_mvir",
        "halo_rvir",
    )
    n_output_mock = galsampler_res.target_gals_target_halo_ids.size
    idxA, idxB = crossmatch(galsampler_res.target_gals_target_halo_ids, unit["halo_id"])
    output_mock = Table()
    for key in keys_to_inherit_from_unit:
        output_mock["unit_" + key] = np.zeros(n_output_mock)
        output_mock["unit_" + key][idxA] = unit[key][idxB]
    t7 = time()
    print("{0:.1f} seconds to inherit from unit with crossmatch".format(t7 - t6))

    # Inherit from TNG
    keys_to_inherit_from_tng = list(tng.keys())
    for key in keys_to_inherit_from_tng:
        output_mock["tng_" + key] = tng[key][galsampler_res.target_gals_selection_indx]

    _key = "galsampler_target_halo_ids"
    output_mock[_key] = galsampler_res.target_gals_target_halo_ids

    _key = "galsampler_source_halo_ids"
    output_mock[_key] = galsampler_res.target_gals_source_halo_ids
    t8 = time()
    print("{0:.1f} seconds to inherit from TNG".format(t8 - t7))

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
    is_sat_target = ~output_mock["tng_is_central"]

    is_sat_source = ~tng["is_central"]
    logmh_host_source = tng["host_halo_logmh"]
    pos_host_source = tng["host_halo_pos"]
    vel_host_source = tng["host_halo_vel"]
    pos_source = tng["subhalo_pos"]
    vel_source = tng["subhalo_vel"]
    pos_target, vel_target = inherit_host_centric_posvel(
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
    )
    output_mock["pos"] = np.mod(pos_target, UNIT_LBOX)
    output_mock["vel"] = vel_target

    is_cen_target = output_mock["tng_is_central"]
    vel_source = output_mock["tng_subhalo_vel"]
    vel_host_source = output_mock["tng_host_halo_vel"]
    vel_host_target = output_mock["vel"]
    output_mock["vel"] = add_central_velbias(
        is_cen_target, vel_source, vel_host_source, vel_host_target
    )
    t9 = time()
    print("{0:.1f} seconds to assign phase space".format(t9 - t8))

    tng_phot_sample_fn = "/lcrc/project/halotools/C3EMC/TNG300-1/tng_phot_sample.h5"
    tng_phot_sample = Table.read(tng_phot_sample_fn, path="data")
    tng_phot_sample["logsm"] = np.log10(tng_phot_sample["SubhaloMassType"][:, 4]) + 10

    tng_phot_sample["lgssfr"] = compute_lg_ssfr(
        10 ** tng_phot_sample["logsm"], tng_phot_sample["SubhaloSFR"]
    )

    source_props = (tng_phot_sample["logsm"], tng_phot_sample["lgssfr"])
    target_props = (
        np.log10(output_mock["tng_mstar"]),
        output_mock["tng_lgssfr"],
    )

    dd_match, indx_match = galmatch.calculate_indx_correspondence(
        source_props, target_props
    )
    output_mock["tng_grizy"] = tng_phot_sample["grizy"][indx_match]
    t10 = time()
    print("{0:.1f} seconds to inherit DESI photometry".format(t10 - t9))

    print("\n")
    print(output_mock.keys())
    outname = os.path.join(OUTDRN, args.outname)
    output_mock.write(outname, path="data", overwrite=True)
    print("\n")
