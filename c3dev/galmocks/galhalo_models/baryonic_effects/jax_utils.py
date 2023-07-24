"""
"""
from jax import jit as jjit
from jax import numpy as jnp


@jjit
def _tw_cuml_kern(x, m, h):
    """Triweight kernel version of an err function."""
    z = (x - m) / h
    a = +1 / 2
    b = +35 * z / 96
    c = -35 * z**3 / 864
    d = +7 * z**5 / 2592
    e = -5 * z**7 / 69984
    val = a + b + c + d + e
    val = jnp.where(z < -3, 0, val)
    val = jnp.where(z > 3, 1, val)
    return val


@jjit
def _tw_sigmoid(x, x0, dx, ymin, ymax):
    """Triweight kernel version of a sigmoid."""
    tw_h = dx / 6
    height_diff = ymax - ymin
    body = _tw_cuml_kern(x, x0, tw_h)
    return ymin + height_diff * body


@jjit
def _double_tw_sigmoid(
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
    h_trans,
):
    func_lo = _tw_sigmoid(x, x0_lo, tw_h_lo, ymin_lo, ymax_lo)
    func_hi = _tw_sigmoid(x, x0_hi, tw_h_hi, ymin_hi, ymax_hi)
    func = _tw_sigmoid(x, x0_trans, h_trans, func_lo, func_hi)
    return func
