"""
"""
import numpy as np


def compute_cumulative_number_density(x, v):
    sorted_cuml_nd = np.arange(1, len(x) + 1) / v
    indx_x_is_sorted = np.argsort(x)
    cuml_nd = np.zeros_like(x)
    cuml_nd[indx_x_is_sorted] = sorted_cuml_nd
    return cuml_nd, indx_x_is_sorted


def get_abunmatched_quantity(lgx1, lgx2, v1, v2, reverse=False):
    if reverse:
        lgx1 = -lgx1
        lgx2 = -lgx2
    cuml_nd1, indx_x1 = compute_cumulative_number_density(lgx1, v1)
    cuml_nd2, indx_x2 = compute_cumulative_number_density(lgx2, v2)

    xt = np.log10(cuml_nd2[indx_x2])
    yt = lgx2[indx_x2]
    lgx1_out = np.interp(np.log10(cuml_nd1), xt, yt)
    if reverse:
        lgx1_out = -lgx1_out
    return lgx1_out
