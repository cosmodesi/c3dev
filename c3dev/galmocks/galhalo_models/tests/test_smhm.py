"""
"""
import numpy as np
from scipy.stats import binned_statistic
from ..smhm import mc_logsm, DEFAULT_SMHM_PARAMS


RNG_SEED = 43


def test_mc_logsm_returns_reasonable_results_with_default_arguments():
    smhm_params = np.array(list(DEFAULT_SMHM_PARAMS.values()))
    n_h = int(1e5)
    logmh = np.linspace(11, 15, n_h)
    rng = np.random.RandomState(RNG_SEED)
    percentile = rng.uniform(0, 1, n_h)
    scatter_level = 0.2
    logsm = mc_logsm(smhm_params, logmh, percentile, scatter_level)
    assert logsm.shape == logmh.shape
    assert np.all(logsm > 5)
    assert np.all(logsm < 13)

    n_bins = 20
    logmh_bins = np.linspace(logmh.min(), logmh.max(), n_bins)
    median_logsm, __, __ = binned_statistic(
        logmh, logsm, bins=logmh_bins, statistic="median"
    )
    assert np.all(np.diff(median_logsm) > 0)

    std_logsm, __, __ = binned_statistic(logmh, logsm, bins=logmh_bins, statistic="std")
    assert np.allclose(std_logsm, scatter_level, atol=0.05)
