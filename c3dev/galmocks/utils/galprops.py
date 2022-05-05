"""
"""
import numpy as np


def compute_lg_ssfr(mstar, sfr, lgssfr_q=-11.8, low_ssfr_cut=1e-12):
    raw_ssfr = sfr / mstar
    ssfr_quenched = 10 ** np.random.normal(loc=lgssfr_q, scale=0.35, size=len(raw_ssfr))
    ssfr = np.where(raw_ssfr < low_ssfr_cut, ssfr_quenched, raw_ssfr)
    return np.log10(ssfr)
