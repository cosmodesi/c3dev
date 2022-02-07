"""
"""
from collections import OrderedDict
import numpy as np
from halotools.utils import sliding_conditional_percentile
from jax import random as jran
from scipy.stats import norm


DEFAULT_SMHM_PARAMS = OrderedDict(
    smhm_logm_crit=11.35,
    smhm_ratio_logm_crit=-1.65,
    smhm_k_logm=1.6,
    smhm_lowm_index_x0=11.5,
    smhm_lowm_index_k=2,
    smhm_lowm_index_ylo=2.5,
    smhm_lowm_index_yhi=2.5,
    smhm_highm_index_x0=13.5,
    smhm_highm_index_k=2,
    smhm_highm_index_ylo=0.5,
    smhm_highm_index_yhi=0.5,
)

DEFAULT_SMHM_SCATTER = 0.2


def _get_cen_sat_percentile(x, y, cenmsk, nwin, ran_key):
    n_gals = cenmsk.size
    n_cens = cenmsk.sum()
    n_sats = n_gals - n_cens

    p_cens = sliding_conditional_percentile(x[cenmsk], y[cenmsk], nwin)
    p_sats = jran.uniform(ran_key, shape=(n_sats,))

    percentile = np.zeros(n_gals)
    percentile[cenmsk] = p_cens
    percentile[~cenmsk] = p_sats

    return percentile


def mc_logsm(smhm_params, logmh, p, scatter):
    median_logsm = _logsm_from_logmh(smhm_params, logmh)
    logsm = norm.isf(1 - p, loc=median_logsm, scale=scatter)
    return logsm


def _logsm_from_logmh(smhm_params, logmh):
    """Kernel of the three-roll SMHM mapping Mhalo ==> Mstar.

    Parameters
    ----------
    smhm_params : ndarray, shape (11, )
        Parameters of the three-roll SMHM used to map Mhalo ==> Mstar,

    logmh : ndarray, shape (n, )
        Base-10 log of halo mass

    Returns
    -------
    logsm : ndarray, shape (n, )
        Base-10 log of stellar mass

    """
    logm_crit, log_sfeff_at_logm_crit, smhm_k_logm = smhm_params[0:3]
    lo_indx_pars = smhm_params[3:7]
    hi_indx_pars = smhm_params[7:11]

    lowm_index = _sigmoid(logmh, *lo_indx_pars)
    highm_index = _sigmoid(logmh, *hi_indx_pars)

    logsm_at_logm_crit = logm_crit + log_sfeff_at_logm_crit
    powerlaw_index = _sigmoid(logmh, logm_crit, smhm_k_logm, lowm_index, highm_index)

    return logsm_at_logm_crit + powerlaw_index * (logmh - logm_crit)


def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + np.exp(-k * (x - x0)))
