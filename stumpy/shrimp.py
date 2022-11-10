import logging

import numpy as np
from numba import njit, prange
import numba

from . import core, config

logger = logging.getLogger(__name__)


@njit(fastmath=True)
def insertsorted_aspiration(aspirations, aspirations_ij, D, i, j):
    """
    Insert distance in aspiration queue.
    Parameters
    ----------
    aspirations : numpy.ndarray
        Priority queue of aspiration distances
    aspirations_ij : numpy.ndarray
        2-column array of `i, j` indices corresponding to `aspirations` distances
    D : float
        Distance to be inserted
    i : int
        Index of time series A corresponding to distance `D`
    j : int
        Index of time series B corresponding to distance `D`
    """
    # get insertion index of distance
    idx = np.searchsorted(aspirations, D)
    # if distance larger than all current distances, do nothing
    if idx > aspirations.shape[0] - 1:
        return

    # shift distances and indices
    aspirations[idx + 1 :] = aspirations[idx:-1]
    aspirations_ij[idx + 1 :, :] = aspirations_ij[idx:-1, :]

    # insert distance and its corresponding indices
    aspirations[idx] = D
    aspirations_ij[idx][0] = i
    aspirations_ij[idx][1] = j


@njit(fastmath=True)
def hill_climb_diagonal(
    T_A,
    T_B,
    m,
    i,
    j,
    dot,
    diagonally_down,
    P,
    I,
):
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]

    if diagonally_down:
        step = 1
    else:
        step = -1

    _i = i + step
    _j = j + step
    _dot = dot
    prev_d = np.inf

    while (diagonally_down and _i < n_A - m + 1 and _j < n_B - m + 1) or (
        (not diagonally_down) and _i >= 0 and _j >= 0
    ):
        mu_T_A = np.mean(T_A[_i : _i + m])
        sigma_T_A = np.std(T_A[_i : _i + m])
        mu_T_B = np.mean(T_B[_j : _j + m])
        sigma_T_B = np.std(T_B[_j : _j + m])

        if diagonally_down:
            _dot = (
                _dot - (T_A[_i - 1] * T_B[_j - 1]) + (T_A[_i + m - 1] * T_B[_j + m - 1])
            )
        else:
            _dot = _dot + (T_A[_i] * T_B[_j]) - (T_A[_i + m] * T_B[_j + m])

        f = (_dot - (m * mu_T_A * mu_T_B)) / (m * sigma_T_A * sigma_T_B)
        d = np.sqrt(2 * m * (1 - f))

        if d < P[_i]:
            P[_i] = d
            I[_i] = _j
        elif prev_d < d:
            break

        prev_d = d
        _i += step
        _j += step


@njit(fastmath=True, parallel=True)
def _shrimp(
    T_A,
    T_B,
    m,
    mu_Q,
    sigma_Q,
    mu_T,
    sigma_T,
    s,
    max_aspirations,
    excl_zone=None,
):
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1

    if excl_zone is None:
        j_init = -l
    else:
        j_init = excl_zone

    n_threads = numba.config.NUMBA_NUM_THREADS
    per_thread_load = max_aspirations // n_threads

    P = np.full(l, np.inf, dtype=np.float64)
    I = np.zeros(l, dtype=np.uint64)
    aspirations = np.full(max_aspirations, np.inf, dtype=np.float64)
    aspirations_ij = np.zeros((max_aspirations, 2), dtype=np.uint64)
    iter_range = np.arange(max(j_init, s), n_B - m + 1 - s, s)

    for idx in prange(iter_range.shape[0]):
        j = iter_range[idx]
        mass_len = min(l, j - j_init)
        Q = T_B[j : j + m]
        T = T_A[:mass_len + m - 1]

        QT = core._sliding_dot_product(Q, T)
        D = core._mass(Q, T, QT, mu_Q[j], sigma_Q[j], mu_T, sigma_T)

        for i in range(l):
            if D[i] < P[i]:
                P[i] = D[i]
                I[i] = j

            insertsorted_aspiration(aspirations, aspirations_ij, D[i], i, j)

    m_uint = np.uint64(m)
    for pid in prange(n_threads):
        for aspiration_idx in range(
            pid * per_thread_load, (pid * per_thread_load) + per_thread_load
        ):
            D = aspirations[aspiration_idx]
            i, j = aspirations_ij[aspiration_idx]

            if np.isinf(D):
                break

            dot = np.dot(T_A[i : i + m_uint], T_B[j : j + m_uint])

            hill_climb_diagonal(
                T_A,
                T_B,
                m_uint,
                i,
                j,
                dot,
                True,
                P,
                I,
            )

            hill_climb_diagonal(
                T_A,
                T_B,
                m_uint,
                i,
                j,
                dot,
                False,
                P,
                I,
            )

    return P, I


def shrimp(T_A, m, T_B=None, ignore_trivial=True, s=None, max_aspirations=None):
    if T_B is None:
        T_B = T_A
        ignore_trivial = True
        excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    T_A, mu_T, sigma_T = core.preprocess(T_A, m)
    T_B, mu_Q, sigma_Q = core.preprocess(T_B, m)

    if T_A.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T_A is {T_A.ndim}-dimensional and must be 1-dimensional. "
            "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
        )

    if T_B.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T_B is {T_B.ndim}-dimensional and must be 1-dimensional. "
            "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
        )

    core.check_window_size(m, max_size=min(T_A.shape[0], T_B.shape[0]))

    if ignore_trivial is False and core.are_arrays_equal(T_A, T_B):  # pragma: no cover
        logger.warning("Arrays T_A, T_B are equal, which implies a self-join.")
        logger.warning("Try setting `ignore_trivial = True`.")

    if ignore_trivial and core.are_arrays_equal(T_A, T_B) is False:  # pragma: no cover
        logger.warning("Arrays T_A, T_B are not equal, which implies an AB-join.")
        logger.warning("Try setting `ignore_trivial = False`.")

    n_A = T_A.shape[0]
    l = n_A - m + 1

    if ignore_trivial:
        excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    else:
        excl_zone = None

    if s is None:
        s = int(m * 2)

    if max_aspirations is None:
        max_aspirations = int(np.sqrt(l)) * 2

    P, I = _shrimp(
        T_A,
        T_B,
        m,
        mu_Q,
        sigma_Q,
        mu_T,
        sigma_T,
        s,
        max_aspirations,
        excl_zone,
    )

    return P, I
