"""
"""
import numpy as np

from ..ds_kernels import DEFAULT_PARAMS, _baryonic_effect_kern


def test_baryonic_effect_kern():
    lgrarr = np.linspace(-1, 1.5, 50)
    res = _baryonic_effect_kern(lgrarr, *DEFAULT_PARAMS)
    assert np.all(np.isfinite(res))
    assert res.shape == lgrarr.shape
