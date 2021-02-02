import numpy as np
import numpy.testing as npt
import stumpy
from dask.distributed import Client, LocalCluster
from numba import cuda
import pytest


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    yield cluster
    cluster.close()


test_data = [
    (np.array([[584, -11, 23, 79, 1001, 0, -19]], dtype=np.float64), 3),
    (np.random.uniform(-1000, 1000, [5, 20]).astype(np.float64), 5),
]


@pytest.mark.parametrize("T, m", test_data)
def test_stump(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    ref = stumpy.aamp(T, m)
    comp = stumpy.stump(T, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_stumped(T, m, dask_cluster):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    with Client(dask_cluster) as dask_client:
        ref = stumpy.aamped(dask_client, T, m)
        comp = stumpy.stumped(dask_client, T, m, normalize=False)
        npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_gpu_stump(T, m):
    if not cuda.is_available():
        pytest.skip("Skipping Tests No GPUs Available")

    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    ref = stumpy.gpu_aamp(T, m)
    comp = stumpy.gpu_stump(T, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_stumpi(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    ref_stream = stumpy.aampi(T, m)
    comp_stream = stumpy.stumpi(T, m, normalize=False)
    for i in range(10):
        t = np.random.rand()
        ref_stream.update(t)
        comp_stream.update(t)
        npt.assert_almost_equal(ref_stream.P_, comp_stream.P_)


def test_ostinato():
    m = 50
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = stumpy.aamp_ostinato(Ts, m)
    comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.ostinato(Ts, m, normalize=False)

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_ostinatoed(dask_cluster):
    m = 50
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    with Client(dask_cluster) as dask_client:
        ref_radius, ref_Ts_idx, ref_subseq_idx = stumpy.aamp_ostinatoed(
            dask_client, Ts, m
        )
        comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.ostinatoed(
            dask_client, Ts, m, normalize=False
        )

        npt.assert_almost_equal(ref_radius, comp_radius)
        npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
        npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


def test_gpu_ostinato():
    if not cuda.is_available():
        pytest.skip("Skipping Tests No GPUs Available")

    m = 50
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = stumpy.gpu_aamp_ostinato(Ts, m)
    comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.gpu_ostinato(
        Ts, m, normalize=False
    )

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


def test_mpdist():
    T_A = np.random.uniform(-1000, 1000, [8]).astype(np.float64)
    T_B = np.random.uniform(-1000, 1000, [64]).astype(np.float64)
    m = 5

    ref = stumpy.aampdist(T_A, T_B, m)
    comp = stumpy.mpdist(T_A, T_B, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


def test_mpdisted(dask_cluster):
    T_A = np.random.uniform(-1000, 1000, [8]).astype(np.float64)
    T_B = np.random.uniform(-1000, 1000, [64]).astype(np.float64)
    m = 5

    with Client(dask_cluster) as dask_client:
        ref = stumpy.aampdisted(dask_client, T_A, T_B, m)
        comp = stumpy.mpdisted(dask_client, T_A, T_B, m, normalize=False)
        npt.assert_almost_equal(ref, comp)


def test_gpu_mpdist():
    if not cuda.is_available():
        pytest.skip("Skipping Tests No GPUs Available")

    T_A = np.random.uniform(-1000, 1000, [8]).astype(np.float64)
    T_B = np.random.uniform(-1000, 1000, [64]).astype(np.float64)
    m = 5

    ref = stumpy.gpu_aampdist(T_A, T_B, m)
    comp = stumpy.gpu_mpdist(T_A, T_B, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump(T, m):
    ref = stumpy.maamp(T, m)
    comp = stumpy.mstump(T, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_mstumped(T, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        ref = stumpy.maamped(dask_client, T, m)
        comp = stumpy.mstumped(dask_client, T, m, normalize=False)
        npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_subspace(T, m):
    motif_idx = 1
    nn_idx = 4

    for k in range(T.shape[0]):
        ref_S = stumpy.maamp_subspace(T, m, motif_idx, nn_idx, k)
        comp_S = stumpy.subspace(T, m, motif_idx, nn_idx, k, normalize=False)
        npt.assert_almost_equal(ref_S, comp_S)