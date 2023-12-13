import logging

import numpy as np
from numba import njit, prange
import numba

from . import core, config

logger = logging.getLogger(__name__)


@njit(fastmath=True)
def insertsorted_distance(distances, indices, D, row, col):
    """
    Insert distance in priority queue. This function can be used to maintain a sorted
    array of distances and the corresponding distance matrix indices.

    Parameters
    ----------
    distances : numpy.ndarray
        Priority queue of distances

    indices : numpy.ndarray
        2-column array of `i, j` indices corresponding to the `distances` values

    D : float
        Distance to be inserted

    row : int
        Row index of distance `D` in the distance matrix

    col : int
        Column index of distance `D` in the distance matrix
    """
    # get insertion index of distance
    idx = np.searchsorted(distances, D)
    # if distance larger than all current distances, do nothing
    if idx > distances.shape[0] - 1:
        return

    # shift distances and indices
    distances[idx + 1 :] = distances[idx:-1]
    indices[idx + 1 :, :] = indices[idx:-1, :]

    # insert distance and its corresponding indices
    distances[idx] = D
    indices[idx][0] = row
    indices[idx][1] = col


@njit(fastmath=True)
def hill_climb_diagonal(T_A, T_B, m, row, col, dot_product, diagonally_down, P, I):
    """
    Perform hill climbing diagonally from the given point and update the matrix profile
    where appropriate.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in `T_A`, its nearest neighbor in `T_B` will be recorded.

    m : int
        Window size

    row : int
        Row index of the starting distance in the distance matrix

    col : int
        Column index of the starting distance in the distance matrix

    dot_product : float
        Dot product of the starting distance

    diagonally_down : bool
        Whether to go diagonally up or down. If this is true, traversal is diagonally
        down and towards the right. If this is false, traversal is diagonally up and
        towards the left.

    P : numpy.ndarray
        The matrix profile distances

    I : numpy.ndarray
        The matrix profile indices
    """
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]

    if diagonally_down:
        step = 1
    else:
        step = -1

    # next set of indices
    i = row + step
    j = col + step
    # current dot product
    dot = dot_product
    # previous distance
    prev_d = np.inf

    while (diagonally_down and i < n_A - m + 1 and j < n_B - m + 1) or (
        (not diagonally_down) and i >= 0 and j >= 0
    ):
        # compute mean and std
        mu_T_A = np.mean(T_A[i : i + m])
        sigma_T_A = np.std(T_A[i : i + m])
        mu_T_B = np.mean(T_B[j : j + m])
        sigma_T_B = np.std(T_B[j : j + m])

        # get dot product of current indices using dot product of previous indices
        if diagonally_down:
            dot = dot - (T_A[i - 1] * T_B[j - 1]) + (T_A[i + m - 1] * T_B[j + m - 1])
        else:
            dot = dot + (T_A[i] * T_B[j]) - (T_A[i + m] * T_B[j + m])

        # compute distance using dot product
        f = (dot - (m * mu_T_A * mu_T_B)) / (m * sigma_T_A * sigma_T_B)
        d = np.sqrt(2 * m * (1 - f))

        # if current distance is the shortest computed so far, update matrix profile
        if d < P[i]:
            P[i] = d
            I[i] = j
        # otherwise, if the distance went up since the previous iteration, break loop
        elif prev_d < d:
            break

        # update distance and indices
        prev_d = d
        i += step
        j += step


@njit(fastmath=True, parallel=True)
def _shrimp(
    T_A,
    T_B,
    m,
    mu_Q,
    mu_T,
    sigma_Q,
    sigma_T,
    s,
    max_aspirations,
    mass_start_idx,
):
    """
    Perform SHRIMP algorithm using JIT-compilation and multi-threading to compute an
    approximation of the matrix profile.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in `T_A`, its nearest neighbor in `T_B` will be recorded.

    m : int
        Window size

    mu_Q : numpy.ndarray
        Sliding mean of time series `T_B`

    mu_T : numpy.ndarray
        Mean of time series `T_A`

    sigma_Q : numpy.ndarray
        Sliding standard deviation of time series `T_B`

    sigma_T : numpy.ndarray
        Standard deviation of time series `T_A`

    s : int
        Sampling interval for computing MASS by column

    max_aspirations : int
        Maximum number of aspiration distances to store

    mass_start_idx : int
        Starting diagonal index

    Returns
    -------
    P : numpy.ndarray
        The matrix profile distances

    I : numpy.ndarray
        The matrix profile indices
    """
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1
    uint64_m = np.uint64(m)

    n_threads = numba.config.NUMBA_NUM_THREADS

    # matrix profile distances
    P = np.full(l, np.inf, dtype=np.float64)
    # matrix profile indices
    I = np.zeros(l, dtype=np.uint64)

    # array to store lowest found distances, as aspirations
    aspirations = np.full(max_aspirations, np.inf, dtype=np.float64)
    # indices of aspiration distances
    aspiration_indices = np.zeros((max_aspirations, 2), dtype=np.uint64)
    # range of columns on which to perform mass
    mass_range = np.arange(max(mass_start_idx, s), n_B - m + 1 - s, s)
    # number of columns to perform mass on in each thread
    per_thread_load = int(np.ceil(mass_range.shape[0] / n_threads))

    # iterate over threads
    for pid in prange(n_threads):
        # iteration range of each thread
        start_idx = pid * per_thread_load
        end_idx = min(start_idx + per_thread_load, mass_range.shape[0])

        for idx in range(start_idx, end_idx):
            # j is the column to perform mass on
            j = mass_range[idx]
            # length of column to perform mass on
            # normally this is just the entire column, but can be shorter if there is an
            # exclusion zone
            mass_column_len = min(l, j - mass_start_idx + 1)

            # perform mass
            Q = T_B[j : j + m]
            T = T_A[: mass_column_len + m - 1]
            QT = core._sliding_dot_product(Q, T)
            D = core._mass(
                Q,
                T,
                QT,
                mu_Q[j],
                sigma_Q[j],
                mu_T[:mass_column_len],
                sigma_T[:mass_column_len],
            )

            # iterate over distances computed in by mass
            for i in range(D.shape[0]):
                # update matrix profile if this is the shortest distance found so far
                if D[i] < P[i]:
                    P[i] = D[i]
                    I[i] = j

                # insert distance into priority queue
                insertsorted_distance(aspirations, aspiration_indices, D[i], i, j)

    # number of aspiration distances to perform hill climbing from
    per_thread_load = int(np.ceil(max_aspirations / n_threads))

    # iterate over threads
    for pid in prange(n_threads):
        # iteration range of each thread
        start_idx = pid * per_thread_load
        end_idx = min(start_idx + per_thread_load, max_aspirations)

        for aspiration_idx in range(start_idx, end_idx):
            D = aspirations[aspiration_idx]
            i, j = aspiration_indices[aspiration_idx]

            # if distance is infinite, no point adding to priority queue
            if np.isinf(D):
                break

            # compute dot product at these indices
            dot_product = np.dot(T_A[i : i + uint64_m], T_B[j : j + uint64_m])
            # perform hill climbing diagonally down
            hill_climb_diagonal(T_A, T_B, uint64_m, i, j, dot_product, True, P, I)
            # perform hill climbing diagonally up
            hill_climb_diagonal(T_A, T_B, uint64_m, i, j, dot_product, False, P, I)

    return P, I


def shrimp(T_A, m, T_B=None, ignore_trivial=True, s=None, max_aspirations=None):
    """
    Perform SHRIMP algorithm using JIT-compilation and multi-threading to compute an
    approximation of the matrix profile.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    T_B : numpy.ndarray, default None
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in `T_A`, its nearest neighbor in `T_B` will be recorded.

    ignore_trivial : bool, default True
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.

    s : int, default None
        Sampling interval for computing MASS by column. If not specified, this is set to
        `m * 2`.

    max_aspirations : int
        Maximum number of aspiration distances to store. If not specified, this is set
        to `sqrt(n_A - m + 1) * 2`, where `n_A = len(T_A)`.

    Returns
    -------
    P : numpy.ndarray
        The matrix profile distances

    I : numpy.ndarray
        The matrix profile indices
    """
    if T_B is None:
        T_B = T_A
        ignore_trivial = True

    T_A, mu_T, sigma_T = core.preprocess(T_A, m)
    T_B, mu_Q, sigma_Q = core.preprocess(T_B, m)

    if T_A.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T_A is {T_A.ndim}-dimensional and must be 1-dimensional. "
        )

    if T_B.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T_B is {T_B.ndim}-dimensional and must be 1-dimensional. "
        )

    core.check_window_size(m, max_size=min(T_A.shape[0], T_B.shape[0]))

    n_A = T_A.shape[0]
    l = n_A - m + 1

    if ignore_trivial:
        mass_start_idx = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    else:
        mass_start_idx = -l

    if s is None:
        s = int(m * 2)

    if max_aspirations is None:
        max_aspirations = int(np.sqrt(l)) * 2

    P, I = _shrimp(
        T_A,
        T_B,
        m,
        mu_Q,
        mu_T,
        sigma_Q,
        sigma_T,
        s,
        max_aspirations,
        mass_start_idx,
    )

    return P, I
