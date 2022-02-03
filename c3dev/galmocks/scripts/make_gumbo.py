"""
"""
import numpy as np
import argparse
from time import time
from astropy.table import Table
from c3dev.galmocks.data_loaders.load_umachine import DTYPE as UM_DTYPE
from c3dev.galmocks.utils import galmatch
from halotools.utils import crossmatch, sliding_conditional_percentile
from halotools.empirical_models import noisy_percentile


def compute_lg_ssfr(mstar, sfr, lgssfr_q=-11.8, low_ssfr_cut=1e-12):
    raw_ssfr = sfr / mstar
    ssfr_quenched = 10 ** np.random.normal(loc=lgssfr_q, scale=0.35, size=len(raw_ssfr))
    ssfr = np.where(raw_ssfr < low_ssfr_cut, ssfr_quenched, raw_ssfr)
    return np.log10(ssfr)


UM_LOGSM_CUT = 9.0
OUTPUT_LOGSM_CUT = 9.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("unit_sim_fn", help="path to unit sim subhalo catalog")
    parser.add_argument("um_fn", help="path to um sfr catalog")
    parser.add_argument("outname", help="Output fname")
    args = parser.parse_args()

    t0 = time()
    um = Table(np.fromfile(args.um_fn, dtype=UM_DTYPE))
    logsm_msk = um["sm"] > 10 ** UM_LOGSM_CUT
    um = um[logsm_msk]
    t1 = time()
    unit = Table.read(args.unit_sim_fn, path="data")
    t2 = time()
    print("{0:.1f} seconds to load UM".format(t1 - t0))
    print("{0:.1f} seconds to load UNIT".format(t2 - t1))

    um["uber_host_indx"] = galmatch.compute_uber_host_indx(um["upid"], um["id"])
    t3 = time()
    unit["uber_host_indx"] = galmatch.compute_uber_host_indx(
        unit["halo_upid"], unit["halo_id"]
    )
    t4 = time()
    print("{0:.1f} seconds to compute UM hostid".format(t3 - t2))
    print("{0:.1f} seconds to compute UNIT hostid".format(t4 - t3))

    um["uber_host_haloid"] = um["id"][um["uber_host_indx"]]
    um["mhost"] = um["m"][um["uber_host_indx"]]
    um["rvir_host"] = um["r"][um["uber_host_indx"]] / 1000.0
    um["uber_host_pos"] = um["pos"][um["uber_host_indx"]]
    um["host_delta_pos"] = um["pos"] - um["uber_host_pos"]

    unit["uber_host_haloid"] = unit["halo_id"][unit["uber_host_indx"]]

    cenmsk_um = um["uber_host_haloid"] == um["id"]
    cenmsk_unit = unit["uber_host_haloid"] == unit["halo_id"]

    source_galaxies_host_halo_id = um["uber_host_haloid"]
    source_halo_ids = um["id"][cenmsk_um]
    target_halo_ids = unit["halo_id"][cenmsk_unit]
    source_halo_props = (np.log10(um["m"][cenmsk_um]),)
    target_halo_props = (np.log10(unit["halo_mvir"][cenmsk_unit]),)
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

    # Inherit from UM
    keys_to_inherit_from_um = (
        "m",
        "sm",
        "sfr",
        "uber_host_haloid",
        "id",
        "mhost",
        "rvir_host",
        "host_delta_pos",
    )
    for key in keys_to_inherit_from_um:
        output_mock["um_" + key] = um[key][galsampler_res.target_gals_selection_indx]
    output_mock[
        "galsampler_target_halo_ids"
    ] = galsampler_res.target_gals_target_halo_ids
    output_mock[
        "galsampler_source_halo_ids"
    ] = galsampler_res.target_gals_source_halo_ids
    t8 = time()
    print("{0:.1f} seconds to inherit from UM".format(t8 - t7))

    tng_phot_sample_fn = "/lcrc/project/halotools/C3GMC/TNG300-1/tng_phot_sample.h5"
    tng_phot_sample = Table.read(tng_phot_sample_fn, path="data")
    tng_phot_sample["logsm"] = np.log10(tng_phot_sample["SubhaloMassType"][:, 4]) + 10

    output_mock["um_logsm"] = np.log10(output_mock["um_sm"])

    output_mock["lgssfr"] = compute_lg_ssfr(
        10 ** output_mock["um_logsm"], output_mock["um_sfr"] / 1e9
    )
    tng_phot_sample["lgssfr"] = compute_lg_ssfr(
        10 ** tng_phot_sample["logsm"], tng_phot_sample["SubhaloSFR"]
    )

    output_mock["lgssfr_perc"] = sliding_conditional_percentile(
        output_mock["um_logsm"], output_mock["lgssfr"], 201
    )
    tng_phot_sample["lgssfr_perc"] = sliding_conditional_percentile(
        tng_phot_sample["logsm"], tng_phot_sample["lgssfr"], 201
    )
    output_mock["lgssfr_perc_noisy"] = noisy_percentile(output_mock["lgssfr_perc"], 0.9)

    source_props = (tng_phot_sample["logsm"], tng_phot_sample["lgssfr_perc"])
    target_props = (output_mock["um_logsm"], output_mock["lgssfr_perc_noisy"])

    dd_match, indx_match = galmatch.calculate_indx_correspondence(
        source_props, target_props
    )
    output_mock["tng_logsm"] = tng_phot_sample["logsm"][indx_match]
    output_mock["tng_lgssfr"] = tng_phot_sample["lgssfr"][indx_match]
    output_mock["tng_grizy"] = tng_phot_sample["grizy"][indx_match]
    t9 = time()
    print("{0:.1f} seconds to inherit from TNG".format(t9 - t8))

    # Assign positions
    rhalo_ratio = output_mock["unit_halo_rvir"] * output_mock["um_rvir_host"]
    dx = output_mock["um_host_delta_pos"][:, 0] * rhalo_ratio
    dy = output_mock["um_host_delta_pos"][:, 1] * rhalo_ratio
    dz = output_mock["um_host_delta_pos"][:, 2] * rhalo_ratio

    output_mock["galaxy_x"] = output_mock["unit_halo_x"] + dx
    output_mock["galaxy_y"] = output_mock["unit_halo_y"] + dy
    output_mock["galaxy_z"] = output_mock["unit_halo_z"] + dz

    output_mock["galaxy_vx"] = (
        output_mock["unit_halo_vx"] + output_mock["um_host_delta_pos"][:, 3]
    )
    output_mock["galaxy_vy"] = (
        output_mock["unit_halo_vy"] + output_mock["um_host_delta_pos"][:, 4]
    )
    output_mock["galaxy_vz"] = (
        output_mock["unit_halo_vz"] + output_mock["um_host_delta_pos"][:, 5]
    )

    # Reduce mock size
    output_cut_msk = output_mock["um_logsm"] > OUTPUT_LOGSM_CUT
    output_mock = output_mock[output_cut_msk]

    # Write to disk
    output_mock.write(args.outname, path="data")

    t10 = time()
    print("{0:.1f} seconds total runtime".format(t10 - t0))
