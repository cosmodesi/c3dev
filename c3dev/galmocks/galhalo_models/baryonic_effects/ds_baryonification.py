"""
"""
import os

import numpy as np
from jax import jit as jjit
from jax import vmap

from .ds_kernels import _baryonic_effect_kern, _get_clipped_halo_arrays
from .load_tng_ds_fitting_data import load_regularized_tng_fitting_data

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
FIT_DATA_DRN = os.path.join(_THIS_DRNAME, "ds_fit_data")
FITTING_DATA = load_regularized_tng_fitting_data(FIT_DATA_DRN)

_A = (None, 0, 0, 0, 0, 0)
_baryonic_effect_kern_vmap = jjit(vmap(_baryonic_effect_kern, in_axes=_A))


def deltabar_ds(lgrad, redshift, lgmh, halo_percentile):
    _res = _get_clipped_halo_arrays(redshift, lgmh, halo_percentile)
    grid_interp = FITTING_DATA[3]
    redshift, lgmh, halo_percentile = _res
    pop_params = grid_interp(np.array((redshift, lgmh)).T)

    delta_bar = _baryonic_effect_kern_vmap(
        lgrad,
        pop_params[:, 0],
        pop_params[:, 1],
        pop_params[:, 2],
        pop_params[:, 3],
        pop_params[:, 4],
    )
    return delta_bar
