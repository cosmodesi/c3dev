"""
"""
import os

from .load_tng_ds_fitting_data import load_regularized_tng_fitting_data

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
FIT_DATA_DRN = os.path.join(_THIS_DRNAME, "ds_fit_data")
FITTING_DATA = load_regularized_tng_fitting_data(FIT_DATA_DRN)


def deltabar_ds(lgrad, redshift, lgmh, halo_percentile):
    pass
