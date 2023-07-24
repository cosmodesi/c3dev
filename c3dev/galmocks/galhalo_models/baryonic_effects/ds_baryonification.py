"""Baryonification model of ΔΣ(r) used in EMC to approximate TNG
"""
import os

import numpy as np
from jax import jit as jjit
from jax import vmap

from .ds_kernels import (
    _baryonic_effect_kern,
    _get_clipped_halo_arrays,
    _get_delta_bar_dymin_lo,
)
from .load_tng_ds_fitting_data import load_regularized_tng_fitting_data

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
FIT_DATA_DRN = os.path.join(_THIS_DRNAME, "ds_fit_data")
FITTING_DATA = load_regularized_tng_fitting_data(FIT_DATA_DRN)

_A = (None, 0, 0, 0, 0, 0)
_baryonic_effect_kern_vmap = jjit(vmap(_baryonic_effect_kern, in_axes=_A))


def deltabar_ds(lgrad, redshift, lgmh, halo_percentile):
    """C3-EMC baryonification model

    Parameters
    ----------
    lgrad : ndarray, shape (nrad, )
        Base-10 log of halo-centric distance in Mpc/h

    redshift : ndarray, shape (nhalos, )
        redshift of each halo

    lgmh : ndarray, shape (nhalos, )
        Base-10 log of halo mass in Msun/h

    halo_percentile : ndarray, shape (nhalos, )
        Prob(<x | Mhalo), cumulative percentile of any secondary halo property x
        at fixed halo mass Mhalo, so that 0 <= halo_percentile <= 1.

    Returns
    -------
    delta_bar : ndarray, shape (nhalos, nrad)
        Fractional difference in ΔΣ(r) between hydro and gravity-only halos

        delta_bar := ΔΣ_hydro(r)/ΔΣ_dm(r) - 1

        delta_bar is positive at radii r where ΔΣ(r) is larger in hydro than gravity-only
        and negative when ΔΣ in hydro is smaller than gravity-only

    """
    _res = _get_clipped_halo_arrays(redshift, lgmh, halo_percentile)
    grid_interp = FITTING_DATA[3]
    redshift, lgmh, halo_percentile = _res
    pop_params = grid_interp(np.array((redshift, lgmh)).T)

    dymin_lo = _get_delta_bar_dymin_lo(lgmh)
    ymin_lo = pop_params[:, 2]
    ymin_lo = ymin_lo + (halo_percentile - 0.5) * dymin_lo * 2

    delta_bar = _baryonic_effect_kern_vmap(
        lgrad,
        pop_params[:, 0],
        pop_params[:, 1],
        ymin_lo,
        pop_params[:, 3],
        pop_params[:, 4],
    )
    return delta_bar
