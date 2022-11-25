import numpy as np
from stumpy import shrimp


def z_norm(T):
    return (T - np.mean(T)) / np.std(T)


def z_norm_euclid_dist(T_A, T_B):
    return np.linalg.norm(z_norm(T_A) - z_norm(T_B))


def test_SHRIMP_consistency():
    n_A = 100
    n_B = 100
    m = 20

    T_A = np.random.rand(n_A)
    T_B = np.random.rand(n_B)

    P_TEST, I_TEST = shrimp(T_A=T_A, T_B=T_B, m=m, ignore_trivial=False)

    for i, j in enumerate(I_TEST):
        i, j = int(i), int(j)
        assert j >= 0 and j < n_B - m + 1
        assert np.isinf(P_TEST[i]) or np.allclose(
            P_TEST[i], z_norm_euclid_dist(T_A[i : i + m], T_B[j : j + m])
        )
