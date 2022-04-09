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


TNG_LOGSM_CUT = 9.0

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

    __, tng_cens_unit_mvir_sorted = build_cdf_lookup(
        np.log10(unit["halo_mvir"][cenmsk_unit]), npts_lookup_table=ncens_tng
    )
    __, tng_cens_unit_vmax_sorted = build_cdf_lookup(
        np.log10(unit["halo_vmax"][cenmsk_unit]), npts_lookup_table=ncens_tng
    )
    uber_host_unit_mvir = np.zeros(n_tng)
    uber_host_unit_mvir[cenmsk_tng] = tng_cens_unit_mvir
    uber_host_unit_mvir[~cenmsk_tng] = uber_host_unit_mvir[
        tng["host_index"][~cenmsk_tng]
    ]
    tng["uber_host_unit_mvir"] = uber_host_unit_mvir

    source_galaxies_host_halo_id = tng["host_halo_id"]
    source_halo_ids = tng["subhalo_id"]

    target_halo_ids = unit["halo_id"][cenmsk_unit]

    target_halo_props = (
        np.log10(unit["halo_mvir"][cenmsk_unit]),
        np.log10(unit["halo_vmax"][cenmsk_unit]),
    )

    source_halo_props = (tng_cens_unit_mvir, tng_cens_unit_vmax)

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
