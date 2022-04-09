"""Production script for gumbo_v0.1
"""
import numpy as np
import argparse
from time import time
from astropy.table import Table
from c3dev.galmocks.data_loaders.load_tng import load_xmatched_tng_catalog as load_tng
from c3dev.galmocks.utils import galmatch
from halotools.utils import crossmatch, sliding_conditional_percentile
from halotools.empirical_models import noisy_percentile
from halotools.utils.inverse_transformation_sampling import build_cdf_lookup

TNG_LOGSM_CUT = 9.0


def get_abunmatched_quantity(*args):
    raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("unit_sim_fn", help="path to unit sim subhalo catalog")
    parser.add_argument("tng_fn", help="path to tng catalog")
    parser.add_argument("outname", help="Output fname")
    args = parser.parse_args()

    t0 = time()
    tng = load_tng(args.tng_fn)
    logsm_msk = tng["sm"] > 10 ** TNG_LOGSM_CUT
    tng = tng[logsm_msk]
    t1 = time()
    unit = Table.read(args.unit_sim_fn, path="data")
    t2 = time()
    print("{0:.1f} seconds to load UM".format(t1 - t0))
    print("{0:.1f} seconds to load UNIT".format(t2 - t1))

    n_tng = len(tng)

    cenmsk_unit = unit["uber_host_haloid"] == unit["halo_id"]
    cenmsk_tng = tng["uber_host_haloid"] == tng["not_implemented_yet"]

    rematching_tng_keys = "mvir", "vmax"
    rematching_unit_keys = "halo_mvir", "halo_vmax"
    gen = zip(rematching_tng_keys, rematching_unit_keys)
    rematched_vals = [
        get_abunmatched_quantity(tng[tng_key], unit[unit_key])
        for tng_key, unit_key in gen
    ]
    for key, newvals in zip(rematching_tng_keys, rematched_vals):
        tng["unit_" + key] = newvals

    #
    # __, tng_cens_unit_mvir_sorted = build_cdf_lookup(
    #     np.log10(unit["halo_mvir"][cenmsk_unit]), npts_lookup_table=ncens_tng
    # )
    # __, tng_cens_unit_vmax_sorted = build_cdf_lookup(
    #     np.log10(unit["halo_vmax"][cenmsk_unit]), npts_lookup_table=ncens_tng
    # )
    # uber_host_unit_mvir = np.zeros(n_tng)
    # uber_host_unit_mvir[cenmsk_tng] = tng_cens_unit_mvir
    # uber_host_unit_mvir[~cenmsk_tng] = uber_host_unit_mvir[
    #     tng["host_index"][~cenmsk_tng]
    # ]
    # tng["uber_host_unit_mvir"] = uber_host_unit_mvir

    source_galaxies_host_halo_id = tng["host_halo_id"]
    source_halo_ids = tng["subhalo_id"]

    target_halo_ids = unit["halo_id"][cenmsk_unit]

    target_halo_props = (
        np.log10(unit["halo_mvir"][cenmsk_unit]),
        np.log10(unit["halo_vmax"][cenmsk_unit]),
    )

    source_halo_props = (
        np.log10(tng["unit_mvir"][cenmsk_unit]),
        np.log10(tng["unit_vmax"][cenmsk_unit]),
    )

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
    keys_to_inherit_from_tng = (
        "m",
        "sm",
        "sfr",
        "uber_host_haloid",
        "id",
        "mhost",
        "rvir_host",
        "host_delta_pos",
    )
    for key in keys_to_inherit_from_tng:
        output_mock["um_" + key] = tng[key][galsampler_res.target_gals_selection_indx]
    output_mock[
        "galsampler_target_halo_ids"
    ] = galsampler_res.target_gals_target_halo_ids
    output_mock[
        "galsampler_source_halo_ids"
    ] = galsampler_res.target_gals_source_halo_ids
    t8 = time()
    print("{0:.1f} seconds to inherit from TNG".format(t8 - t7))
