"""
"""
import numpy as np

from ..ds_baryonification import FITTING_DATA, deltabar_ds
from ..ds_kernels import DEFAULT_PARAMS, _baryonic_effect_kern


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


def test_deltabar_ds():
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


def test_deltabar_ds_random_halos():
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
    assert np.all(delta_bar[0, :] <= delta_bar[-1, :])

    # At high mass, delta_bar has no conc-dependence
    delta_bar = deltabar_ds(lgrad, redshift, zz + 15, halo_percentile)
    assert np.allclose(delta_bar[0, :], delta_bar[-1, :])
    # assert np.all(delta_bar[0, :] <= delta_bar[-1, :])
