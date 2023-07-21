"""
"""
import numpy as np

from ..ds_baryonification import FITTING_DATA
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
