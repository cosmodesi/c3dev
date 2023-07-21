"""
"""
import os
import typing
from collections import OrderedDict

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from .jax_utils import _double_tw_sigmoid, _tw_sigmoid


class Params(typing.NamedTuple):
    x0_lo: float
    dx_lo: float
    ymin_lo: float
    ymax_lo: float
    x0_trans: float


DEFAULT_PARAMS = Params(-0.1, 10.0, 0.1, -0.01, 0.1)
DEFAULT_PDICT = OrderedDict(
    [(key, val) for key, val in zip(DEFAULT_PARAMS, DEFAULT_PARAMS._fields)]
)

X0_TRANS_C0, X0_TRANS_C1 = 0.15, 0.75


@jjit
def get_x0_trans(x0_lo):
    return X0_TRANS_C0 + X0_TRANS_C1 * x0_lo


@jjit
def _get_delta_bar_dymin_lo(lgm0):
    dymin_lo = _tw_sigmoid(lgm0, 13.0, 7, 0.08, -0.02)
    dymin_lo = jnp.where(dymin_lo < 0.0, 0.0, dymin_lo)
    return dymin_lo


@jjit
def _get_params_scalar(lgm, lgmarr, x0_lo_arr, dx_lo_arr, ymin_lo_arr, ymax_lo_arr):
    x0_lo = jnp.interp(lgm, lgmarr, x0_lo_arr)
    dx_lo = jnp.interp(lgm, lgmarr, dx_lo_arr)
    ymin_lo = jnp.interp(lgm, lgmarr, ymin_lo_arr)
    ymax_lo = jnp.interp(lgm, lgmarr, ymax_lo_arr)
    x0_trans = get_x0_trans(x0_lo)
    return x0_lo, dx_lo, ymin_lo, ymax_lo, x0_trans


_a = (0, *[None] * 5)
_get_params_vmap = jjit(vmap(_get_params_scalar, in_axes=_a))


@jjit
def _get_all_ds_params(lgm, p, lgmarr, x0_lo_arr, dx_lo_arr, ymin_lo_arr, ymax_lo_arr):
    m0_params = _get_params_scalar(
        lgm, lgmarr, x0_lo_arr, dx_lo_arr, ymin_lo_arr, ymax_lo_arr
    )
    ymin_lo_med = m0_params[2]
    dymin_lo = _get_delta_bar_dymin_lo(lgm)
    p_table = jnp.array((0.0, 1.0))
    ymin_lo_table = jnp.array((ymin_lo_med - dymin_lo, ymin_lo_med + dymin_lo))
    ymin_lo = jnp.interp(p, p_table, ymin_lo_table)
    all_ds_params = jnp.array([*m0_params[:2], ymin_lo, *m0_params[3:]])
    return all_ds_params


@jjit
def _baryonic_effect_kern(
    x,
    x0_lo,
    dx_lo,
    ymin_lo,
    ymax_lo,
    x0_trans,
):
    tw_h_lo = dx_lo / 6
    x0_hi = x0_lo
    tw_h_hi = tw_h_lo
    ymin_hi = ymax_lo
    tw_h_hi = tw_h_lo
    tw_h_trans = tw_h_lo
    ymax_hi = 0.0

    res = _double_tw_sigmoid(
        x,
        x0_lo,
        tw_h_lo,
        ymin_lo,
        ymax_lo,
        x0_hi,
        tw_h_hi,
        ymin_hi,
        ymax_hi,
        x0_trans,
        tw_h_trans,
    )
    return res
