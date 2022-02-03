"""
"""
from warnings import warn
import numpy as np
from scipy.spatial import cKDTree
from halotools.utils import crossmatch, compute_richness
from numba import njit
from collections import namedtuple

GalsamplerCorrespondence = namedtuple(
    "GalsamplerCorrespondence",
    [
        "target_gals_selection_indx",
        "target_gals_target_halo_ids",
        "target_gals_source_halo_ids",
    ],
)

__all__ = ("compute_source_galaxy_selection_indices",)


@njit
def galaxy_selection_kernel(first_source_gal_indices, richness, n_target_halo, result):
    """Numba kernel filling in array of galaxy selection indices

    Parameters
    ----------
    first_source_gal_indices : ndarray of shape (n_target_halo, )
        Stores the index of the first galaxy in the galaxy catalog
        assigned to each target halo

    richness : ndarray of shape (n_target_halo, )
        Stores the number of galaxies that will be mapped to each target halo

    n_target_halo : int

    result : ndarray of shape richness.sum()

    """
    cur = 0
    for i in range(n_target_halo):
        ifirst = first_source_gal_indices[i]
        n = richness[i]
        if n > 0:
            ilast = ifirst + richness[i]
            for j in range(ifirst, ilast):
                result[cur] = j
                cur += 1


def _get_data_block(*halo_properties):
    return np.vstack(halo_properties).T


def calculate_halo_correspondence(source_halo_props, target_halo_props, n_threads=-1):
    """Calculating indexing array defined by a statistical correspondence between
    source and target halos.

    Parameters
    ----------
    source_halo_props : sequence of n_props ndarrays
        Each ndarray should have shape (n_source_halos, )

    target_halo_props : sequence of n_props ndarrays
        Each ndarray should have shape (n_target_halos, )

    Returns
    -------
    dd_match : ndarray of shape (n_target_halos, )
        Euclidean distance to the source halo matched to each target halo

    indx_match : ndarray of shape (n_target_halos, )
        Index of the source halo matched to each target halo

    """
    assert len(source_halo_props) == len(target_halo_props)
    X_source = _get_data_block(*source_halo_props)
    X_target = _get_data_block(*target_halo_props)
    source_tree = cKDTree(X_source)
    dd_match, indx_match = source_tree.query(X_target, workers=n_threads)
    return dd_match, indx_match


def compute_source_galaxy_selection_indices(
    source_galaxies_host_halo_id,
    source_halo_ids,
    target_halo_ids,
    source_halo_props,
    target_halo_props,
):
    """Calculate the indexing array that transfers source galaxies to target halos

    Parameters
    ----------
    source_galaxies_host_halo_id : ndarray of shape (n_source_gals, )
        Integer array storing values appearing in source_halo_ids

    source_halo_ids : ndarray of shape (n_source_halos, )

    target_halo_ids : ndarray of shape (n_target_halos, )

    source_halo_props : sequence of n_props ndarrays
        Each ndarray should have shape (n_source_halos, )

    target_halo_props : sequence of n_props ndarrays
        Each ndarray should have shape (n_target_halos, )

    Returns
    -------
    selection_indices : ndarray of shape (n_target_gals, )
        Integer array storing values in the range [0, n_source_gals-1]

    target_galaxy_target_halo_ids : ndarray of shape (n_target_gals, )
        Integer array storing values appearing in target_halo_ids

    target_galaxy_source_halo_ids : ndarray of shape (n_target_gals, )
        Integer array storing values appearing in source_halo_ids

    """
    #  Sort the source galaxies so that members of a common halo are grouped together
    idx_sorted_source_galaxies = np.argsort(source_galaxies_host_halo_id)
    sorted_source_galaxies_host_halo_id = source_galaxies_host_halo_id[
        idx_sorted_source_galaxies
    ]

    #  Calculate the index correspondence array that will undo the sorting at the end
    num_source_gals = len(source_galaxies_host_halo_id)
    idx_unsorted_galaxy_indices = np.arange(num_source_gals).astype("i8")[
        idx_sorted_source_galaxies
    ]

    #  For each source halo, calculate the number of resident galaxies
    source_halos_richness = compute_richness(
        source_halo_ids, sorted_source_galaxies_host_halo_id
    )

    #  For each source halo, calculate the index of its first resident galaxy
    source_halo_sorted_source_galaxies_indices = _galaxy_table_indices(
        source_halo_ids, sorted_source_galaxies_host_halo_id
    )

    #  For each target halo, calculate the index of the associated source halo
    __, source_halo_selection_indices = calculate_halo_correspondence(
        source_halo_props, target_halo_props
    )

    #  For each target halo, calculate the number of galaxies
    target_halo_richness = source_halos_richness[source_halo_selection_indices]
    num_target_gals = np.sum(target_halo_richness)

    #  For each target halo, calculate the halo ID of the associated source halo
    target_halo_source_halo_ids = source_halo_ids[source_halo_selection_indices]

    #  For each target halo, calculate the index of its first resident galaxy
    target_halo_first_sorted_source_gal_indices = (
        source_halo_sorted_source_galaxies_indices[source_halo_selection_indices]
    )

    #  For every target halo, we know the index of the first and last galaxy to select
    #  Calculate an array of shape (num_target_gals, )
    # with the index of each selected galaxy
    n_target_halos = target_halo_ids.size
    sorted_source_galaxy_selection_indices = np.zeros(num_target_gals).astype(int)
    galaxy_selection_kernel(
        target_halo_first_sorted_source_gal_indices.astype("i8"),
        target_halo_richness.astype("i4"),
        n_target_halos,
        sorted_source_galaxy_selection_indices,
    )

    #  For each target galaxy, calculate the halo ID of its source and target halo
    target_gals_target_halo_ids = np.repeat(target_halo_ids, target_halo_richness)
    target_gals_source_halo_ids = np.repeat(
        target_halo_source_halo_ids, target_halo_richness
    )

    #  For each index in the sorted galaxy catalog,
    #  calculate the index of the catalog in its original order
    target_gals_selection_indx = idx_unsorted_galaxy_indices[
        sorted_source_galaxy_selection_indices
    ]

    return GalsamplerCorrespondence(
        target_gals_selection_indx,
        target_gals_target_halo_ids,
        target_gals_source_halo_ids,
    )


def _galaxy_table_indices(source_halo_id, galaxy_host_halo_id):
    """For every halo in the source halo catalog, calculate the index
    in the source galaxy catalog of the first appearance of a galaxy that
    occupies the halo, reserving -1 for source halos with no resident galaxies.

    Parameters
    ----------
    source_halo_id : ndarray
        Numpy integer array of shape (num_halos, )

    galaxy_host_halo_id : ndarray
        Numpy integer array of shape (num_gals, )

    Returns
    -------
    indices : ndarray
        Numpy integer array of shape (num_halos, ).
        All values will be in the interval [-1, num_gals)
    """
    uval_gals, indx_uval_gals = np.unique(galaxy_host_halo_id, return_index=True)
    idxA, idxB = crossmatch(source_halo_id, uval_gals)
    num_source_halos = len(source_halo_id)
    indices = np.zeros(num_source_halos) - 1
    indices[idxA] = indx_uval_gals[idxB]
    return indices.astype(int)


def compute_hostid(upid, haloid):
    cenmsk = upid == -1
    hostid = np.copy(haloid)
    hostid[~cenmsk] = upid[~cenmsk]
    idxA, idxB = crossmatch(hostid, haloid)

    has_match = np.zeros(haloid.size).astype("bool")
    has_match[idxA] = True
    hostid[~has_match] = haloid[~has_match]
    return hostid, idxA, idxB, has_match


def compute_uber_host_indx(
    upid, haloid, max_order=20, fill_val=-99, return_internals=False
):
    hostid, idxA, idxB, has_match = compute_hostid(upid, haloid)
    cenmsk = hostid == haloid

    if len(idxA) != len(haloid):
        msg = "{0} values of upid have no match. Treating these objects as centrals"
        warn(msg.format(len(haloid) - len(idxA)))

    _integers = np.arange(haloid.size).astype(int)
    uber_host_indx = np.zeros_like(haloid) + fill_val
    uber_host_indx[cenmsk] = _integers[cenmsk]

    n_unmatched = np.count_nonzero(uber_host_indx == fill_val)
    counter = 0
    while (n_unmatched > 0) and (counter < max_order):
        uber_host_indx[idxA] = uber_host_indx[idxB]
        n_unmatched = np.count_nonzero(uber_host_indx == fill_val)
        counter += 1

    if return_internals:
        return uber_host_indx, idxA, idxB
    else:
        return uber_host_indx


def calculate_indx_correspondence(source_props, target_props, n_threads=-1):
    """For each target data object, find a closely matching source data object

    Parameters
    ----------
    source_props : list of n_props ndarrays
        Each ndarray should have shape (n_source, )

    target_props : list of n_props ndarrays
        Each ndarray should have shape (n_target, )

    Returns
    -------
    dd_match : ndarray of shape (n_target, )
        Euclidean distance between each target and its matching source object

    indx_match : ndarray of shape (n_target, )
        Index into the source object that is matched to each target

    """
    assert len(source_props) == len(target_props)
    X_source = _get_data_block(*source_props)
    X_target = _get_data_block(*target_props)
    source_tree = cKDTree(X_source)
    dd_match, indx_match = source_tree.query(X_target, workers=n_threads)
    return dd_match, indx_match
