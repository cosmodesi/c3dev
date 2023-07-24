"""
"""
import os
from collections import OrderedDict

import numpy as np
from astropy.table import Table
from halotools.utils import sliding_conditional_percentile
from scipy.interpolate import RegularGridInterpolator

from .ds_kernels import Params

TASSO = "/Users/aphearin/work/DATA/DESI/C3GMC/TNG"
LGMH_BINS_TNG_TESTING_DATA = np.linspace(12.5, 14.25, 5)


def load_tng_megafile(drn=TASSO, bnpat="0d3"):
    bn = "megafile_tng_and_dark_z_{}.fits".format(bnpat)
    rbins = np.load(os.path.join(drn, "ds_bins.npy"))
    lgrbins = np.log10(rbins)

    data = Table.read(os.path.join(drn, bn))

    msun_per_pc_factor = 1e12
    data["DS_TNG"] = data["DS_TNG"] / msun_per_pc_factor
    data["DS_TNG_DARK"] = data["DS_TNG_DARK"] / msun_per_pc_factor
    data["DS_ratio"] = data["DS_TNG"] / data["DS_TNG_DARK"] - 1

    uvals = np.unique(data["SubhaloGrNr"])
    msg = "data has at least one repeated entry for `SubhaloGrNr` column"
    assert len(data) == len(uvals), msg

    data["lgMsub"] = np.log10(data["SubhaloMass"])

    for key in ("M_acc_dyn", "stellar_mass", "gass_mass", "c200c"):
        newkey = "p_" + key
        data[newkey] = sliding_conditional_percentile(data["lgMsub"], data[key], 41)

    return data, lgrbins


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


def get_target_data_lgmh(lgmh_bins, data, sumstat=np.median, dlgm=0.1):
    mmsks = [np.abs(data["lgMsub"] - lgm0) < dlgm for lgm0 in lgmh_bins]
    target_data = [sumstat(data["DS_ratio"][mmsk], axis=0) for mmsk in mmsks]
    return mmsks, target_data


def measure_target_data_unit_testing(
    tng_megafile_0d3,
    tng_megafile_0d5,
    tng_megafile_1d0,
    lgmh_bins=LGMH_BINS_TNG_TESTING_DATA,
):
    target_data_0d3 = get_target_data_lgmh(lgmh_bins, tng_megafile_0d3)[1]
    target_data_0d5 = get_target_data_lgmh(lgmh_bins, tng_megafile_0d5)[1]
    target_data_1d0 = get_target_data_lgmh(lgmh_bins, tng_megafile_1d0)[1]
    target_data = [target_data_0d3, target_data_0d5, target_data_1d0]
    return lgmh_bins, np.array(target_data)
