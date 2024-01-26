"""
"""
import argparse
import os
from time import time

import numpy as np
from astropy.table import Table
from c3dev.galmocks.galhalo_models.baryonic_effects.ds_baryonification import (
    deltabar_ds,
)
from halotools.utils import sliding_conditional_percentile

LCRC_EMC_DRN = "/lcrc/project/halotools/gbeltzmo/EMC"

LOGRBINS = np.linspace(-1.0, 1.2, 15)
LOGRMIDS = 0.5 * (LOGRBINS[:-1] + LOGRBINS[1:])


def write_results_to_disk(fn, dsout):
    with open(fn, "w") as fout:
        fout.write("# r ds\n")
        for r, ds in zip(LOGRMIDS, dsout):
            outline = "{0:.3e} {1:.3e}\n".format(10**r, ds)
            fout.write(outline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_megafile_fname")
    parser.add_argument("gals_ascii_fname")
    parser.add_argument("galsampled_h5_fname")
    parser.add_argument("-drn", default=LCRC_EMC_DRN)
    parser.add_argument("redshift", help="0.5 or 0.8", choices=["0.5", "0.8"])
    args = parser.parse_args()

    start = time()

    drn = args.drn
    ds_megafile_fname = os.path.join(drn, args.ds_megafile_fname)
    gals_ascii_fname = os.path.join(drn, args.gals_ascii_fname)
    galsampled_h5_fname = os.path.join(drn, args.galsampled_h5_fname)
    redshift = float(args.redshift)

    if "_pos1" in os.path.basename(gals_ascii_fname):
        poskey = "pos_model1"
    elif "_pos2" in os.path.basename(gals_ascii_fname):
        poskey = "pos_model2"
    elif "_pos3" in os.path.basename(gals_ascii_fname):
        poskey = "pos_model3"
    else:
        poskey = "pos"

    if "mass_conc" in ds_megafile_fname:
        galsample_pat = "mass_conc"
    else:
        galsample_pat = "mass"

    args = (galsample_pat, poskey, redshift)
    msg = "\n...working on model {0}+{1} at z={2:.1f}"
    print(msg.format(*args))

    print("...Loading ΔΣ")
    X = np.loadtxt(gals_ascii_fname)
    indx = X[:, 0].astype(int)
    ds_table = np.load(ds_megafile_fname)

    print("...Loading mock")
    mock = Table.read(galsampled_h5_fname, path="data")
    mock["diffsky_isLRG"] = mock["diffsky_isLRG"].astype(bool)
    if redshift == 0.5:
        mock["diffsky_isBGS"] = mock["diffsky_app_mag_r"] < 19.5
    else:
        mock["diffsky_isBGS"] = False

    msg = "mismatched galaxy/halos"
    assert np.allclose(mock["diffsky_is_cen"][indx], X[:, 1]), msg
    assert np.allclose(mock[poskey][indx, 0], X[:, 3], atol=0.01), msg
    assert np.allclose(mock[poskey][indx, 1], X[:, 4], atol=0.01), msg
    assert np.allclose(mock[poskey][indx, 2], X[:, 5], atol=0.01), msg

    print("...Computing concentration percentile")
    mock["diffsky_is_cen"] = mock["diffsky_is_cen"].astype(bool)
    mock["conc_percentile"] = np.random.uniform(0, 1, len(mock))
    cens = mock[mock["diffsky_is_cen"]]
    cens["conc_percentile"] = sliding_conditional_percentile(
        cens["unit_halo_mvir"], cens["unit_halo_nfw_conc"], 201
    )
    mock["conc_percentile"][mock["diffsky_is_cen"]] = cens["conc_percentile"]

    print("...Computing baryonified lensing")
    mock = mock[indx]

    dbar = deltabar_ds(
        LOGRMIDS, redshift, np.log10(mock["unit_halo_mvir"]), mock["conc_percentile"]
    )
    mock["ds_bar"] = ds_table * (1 + dbar)
    mock["ds_bar_no_baryons"] = ds_table

    ds_sample_lrg = np.mean(mock["ds_bar"][mock["diffsky_isLRG"]], axis=0)
    ds_sample_lrg_no_baryons = np.mean(
        mock["ds_bar_no_baryons"][mock["diffsky_isLRG"]], axis=0
    )

    drnout = os.path.join(drn, "gg_lensing")

    bnpat = "lrg_delta_sigma_{0}_{1}_{2:.1f}.txt"
    bnout = bnpat.format(galsample_pat, poskey, redshift)
    fnout = os.path.join(drnout, bnout)
    write_results_to_disk(fnout, ds_sample_lrg)

    bnpat = "lrg_delta_sigma_no_baryons_{0}_{1}_{2:.1f}.txt"
    bnout = bnpat.format(galsample_pat, poskey, redshift)
    fnout = os.path.join(drnout, bnout)
    write_results_to_disk(fnout, ds_sample_lrg_no_baryons)

    if redshift == 0.5:
        ds_sample_bgs = np.mean(mock["ds_bar"][mock["diffsky_isBGS"]], axis=0)
        ds_sample_bgs_no_baryons = np.mean(
            mock["ds_bar_no_baryons"][mock["diffsky_isBGS"]], axis=0
        )

        bnpat = "bgs_delta_sigma_{0}_{1}_{2:.1f}.txt"
        bnout = bnpat.format(galsample_pat, poskey, redshift)
        fnout = os.path.join(drnout, bnout)
        write_results_to_disk(fnout, ds_sample_bgs)

        bnpat = "bgs_delta_sigma_no_baryons_{0}_{1}_{2:.1f}.txt"
        bnout = bnpat.format(galsample_pat, poskey, redshift)
        fnout = os.path.join(drnout, bnout)
        write_results_to_disk(fnout, ds_sample_bgs_no_baryons)

    end = time()
    print("runtime = {0:.1f} seconds\n".format(end - start))
