"""
"""
import os
import typing
from collections import OrderedDict

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .ds_kernels import Params

TASSO = "/Users/aphearin/work/DATA/DESI/C3GMC/TNG"


def load_tng_data(drn):
    bnpat_list = ["0d3", "0d5", "1d0"]
    pd03 = OrderedDict()
    pd05 = OrderedDict()
    pd10 = OrderedDict()
    param_list = [pd03, pd05, pd10]

    for bnpat, param_collector in zip(bnpat_list, param_list):
        lgm0_table = np.loadtxt(os.path.join(drn, bnpat, "lgm0_table.txt"))
        x0_lo_table = np.loadtxt(os.path.join(drn, bnpat, "x0_lo_table.txt"))
        dx_lo_table = np.loadtxt(os.path.join(drn, bnpat, "dx_lo_table.txt"))
        ymin_lo_table = np.loadtxt(os.path.join(drn, bnpat, "ymin_lo_table.txt"))
        ymax_lo_table = np.loadtxt(os.path.join(drn, bnpat, "ymax_lo_table.txt"))
        x0_trans_table = np.loadtxt(os.path.join(drn, bnpat, "x0_trans_table.txt"))
        params_table = Params(
            x0_lo_table,
            dx_lo_table,
            ymin_lo_table,
            ymax_lo_table,
            x0_trans_table,
        )
        param_collector["lgm0_table"] = lgm0_table
        param_collector["params_table"] = params_table

    return (pd03, pd05, pd10)


def regularize_fitting_data(pd03, pd10):
    newpd10 = pd10.copy()
    n_missing = pd03["params_table"].x0_lo.size - pd10["params_table"].x0_lo.size
    for __ in range(n_missing):
        newpd10["params_table"] = Params(
            *[np.append(x, x[-1]) for x in newpd10["params_table"]]
        )
    newpd10["lgm0_table"] = pd03["lgm0_table"]

    return newpd10


def load_regularized_tng_fitting_data(drn):
    pd03, pd05, pd10 = load_tng_data(drn)
    pd10 = regularize_fitting_data(pd03, pd10)
    n_params = len(pd03["params_table"]._fields)
    n_mass = pd03["lgm0_table"].size

    M03 = np.zeros((n_mass, n_params))
    for ip, param in enumerate(pd03["params_table"]):
        M03[:, ip] = param

    M05 = np.zeros((n_mass, n_params))
    for ip, param in enumerate(pd05["params_table"]):
        M05[:, ip] = param

    M10 = np.zeros((n_mass, n_params))
    for ip, param in enumerate(pd10["params_table"]):
        M10[:, ip] = param

    n_redshift = 3
    fitting_data_matrix = np.zeros((n_redshift, n_mass, n_params))
    fitting_data_matrix[0, :, :] = M03
    fitting_data_matrix[1, :, :] = M05
    fitting_data_matrix[2, :, :] = M10

    lgm0_grid = pd03["lgm0_table"]
    redshift_grid = np.array((0.3, 0.5, 1.0))

    grid_interp = get_scipy_param_interpolator(
        redshift_grid, lgm0_grid, fitting_data_matrix
    )
    return redshift_grid, lgm0_grid, fitting_data_matrix, grid_interp, pd03, pd05, pd10


def get_scipy_param_interpolator(redshift_grid, lgm0_grid, fitting_data_matrix):
    points = (redshift_grid, lgm0_grid)
    values = fitting_data_matrix
    grid_interp = RegularGridInterpolator(points, values, fill_value=None)
    return grid_interp
