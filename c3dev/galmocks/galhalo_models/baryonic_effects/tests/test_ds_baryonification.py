"""
"""
import os

import numpy as np
import pytest

from ..ds_baryonification import FITTING_DATA, deltabar_ds
from ..ds_kernels import DEFAULT_PARAMS, _baryonic_effect_kern
from ..load_tng_ds_fitting_data import (
    TASSO,
    load_tng_megafile,
    measure_target_data_unit_testing,
)

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DRN = os.path.join(_THIS_DRNAME, "testing_data")

try:
    assert os.path.isdir(TASSO)
    IS_TASSO_MACHINE = True
except AssertionError:
    IS_TASSO_MACHINE = False


def test_fitting_data():
    (
        redshift_grid,
        lgm0_grid,
        fitting_data_matrix,
        grid_interp,
        pd03,
        pd05,
        pd10,
    ) = FITTING_DATA

    assert np.allclose(redshift_grid, np.array((0.3, 0.5, 1.0)))

    n_redshift = redshift_grid.size
    n_mass = lgm0_grid.size
    n_params = len(DEFAULT_PARAMS)
    assert fitting_data_matrix.shape == (n_redshift, n_mass, n_params)

    z0_arr = np.zeros_like(lgm0_grid) + redshift_grid[0]
    pop_params = grid_interp(np.array((z0_arr, lgm0_grid)).T)

    n_rad = 50
    lgrarr = np.linspace(-1, 1.5, n_rad)
    for im in range(n_mass):
        delta_bar = _baryonic_effect_kern(lgrarr, *pop_params[im, :])
        assert np.all(np.isfinite(delta_bar))
        assert delta_bar.shape == (n_rad,)

        assert np.all(delta_bar) > -1.0
        assert np.all(delta_bar) < 1.0


def test_deltabar_ds_has_expected_bounds():
    nrad, nhalos = 40, 20_000
    lgrad = np.linspace(-1, 1.5, nrad)
    redshift = np.zeros(nhalos) + 0.5
    lgmh = np.zeros(nhalos) + 12.0
    halo_percentile = np.zeros(nhalos) + 0.5

    delta_bar = deltabar_ds(lgrad, redshift, lgmh, halo_percentile)
    assert np.all(np.isfinite(delta_bar))
    assert delta_bar.shape == (nhalos, nrad)
    assert np.all(delta_bar > -1.0)
    assert np.all(delta_bar < 1.0)


def test_deltabar_ds_has_expected_bounds_random_halos():
    nrad, nhalos = 40, 20_000
    lgrad = np.linspace(-1, 1.5, nrad)
    redshift = np.random.uniform(0, 10, nhalos)
    lgmh = np.random.uniform(10, 16, nhalos)
    halo_percentile = np.random.uniform(0, 1, nhalos)

    delta_bar = deltabar_ds(lgrad, redshift, lgmh, halo_percentile)
    assert np.all(np.isfinite(delta_bar))
    assert delta_bar.shape == (nhalos, nrad)
    assert np.all(delta_bar > -1.0)
    assert np.all(delta_bar < 1.0)


def test_deltabar_ds_has_expected_conc_dependence_for_tng_calibration():
    nrad, nhalos = 40, 20_000
    lgrad = np.linspace(-1, 1.5, nrad)
    zz = np.zeros(nhalos)
    redshift = zz + 0.5
    lgmh = zz + 12.0
    halo_percentile = np.linspace(0, 1, nhalos) + 0.5

    # At low mass, delta_bar should be stronger for high-conc halos than low-conc
    delta_bar = deltabar_ds(lgrad, redshift, lgmh, halo_percentile)
    assert not np.allclose(delta_bar[0, :], delta_bar[-1, :])
    assert np.all(delta_bar[0, :] >= delta_bar[-1, :])

    # At high mass, delta_bar has no conc-dependence
    delta_bar = deltabar_ds(lgrad, redshift, zz + 15, halo_percentile)
    assert np.allclose(delta_bar[0, :], delta_bar[-1, :])


@pytest.mark.skipif("not IS_TASSO_MACHINE")
def test_frozen_target_data_is_reproducible():
    tng_0d3, lgrbins = load_tng_megafile(bnpat="0d3")
    tng_0d5, lgrbins = load_tng_megafile(bnpat="0d5")
    tng_1d0, lgrbins = load_tng_megafile(bnpat="1d0")

    lgrmids = 0.5 * (lgrbins[:-1] + lgrbins[1:])
    lgmh_bins_recomputed, target_data_recomputed = measure_target_data_unit_testing(
        tng_0d3, tng_0d5, tng_1d0
    )
    lgmh_bins_frozen = np.loadtxt(os.path.join(TEST_DATA_DRN, "lgmh_bins.txt"))
    tng_ds_target_data_frozen = np.load(
        os.path.join(TEST_DATA_DRN, "tng_ds_target_data.npy")
    )

    lgrmids_frozen = np.loadtxt(os.path.join(TEST_DATA_DRN, "lgrmids.txt"))
    assert np.allclose(lgrmids_frozen, lgrmids), "lgrmids_frozen changed"
    assert np.allclose(lgmh_bins_frozen, lgmh_bins_recomputed), "lgmh_bins changed"

    msg = "target TNG DS data changed"
    assert np.allclose(tng_ds_target_data_frozen, target_data_recomputed), msg


def test_deltabar_ds_model_agrees_with_frozen_target_data():
    lgmh_bins = np.loadtxt(os.path.join(TEST_DATA_DRN, "lgmh_bins.txt"))
    tng_ds_target_data = np.load(os.path.join(TEST_DATA_DRN, "tng_ds_target_data.npy"))
    ztestarr = np.array((0.3, 0.5, 1.0))
    lgrmids = np.loadtxt(os.path.join(TEST_DATA_DRN, "lgrmids.txt"))

    TOL = 0.05
    xx = np.zeros(1)
    msg = "at z={0:.2f} MSE loss disagrees with frozen TNG target data"
    for iz, ztest in enumerate(ztestarr):
        for im in range(lgmh_bins.size):
            delta_bar_target = tng_ds_target_data[iz, im, :]

            delta_bar_pred = deltabar_ds(
                lgrmids, xx + ztest, xx + lgmh_bins[im], xx + 0.5
            )
            delta_bar_pred = delta_bar_pred.flatten()
            loss = np.sqrt(_mse(delta_bar_pred, delta_bar_target))
            assert loss < TOL, msg.format(ztest, loss)


def _mse(pred, target):
    diff = pred - target
    return np.mean(diff**2)
