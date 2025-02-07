import numpy as np
import jax

import riecovest.distance as dist

def get_random_pd_mat(mat_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    rank = 2 * mat_size
    R = np.zeros((mat_size, mat_size), dtype=complex)
    for r in range(rank):
        v = rng.normal(size=(mat_size, 1)) + 1j*rng.normal(size=(mat_size, 1))
        R += v @ v.conj().T

    assert np.allclose(R, R.conj().T)
    return R

def test_gevd_distance_with_full_rank_equals_old_implementation():
    dim = 5
    A = get_random_pd_mat(dim)
    B = get_random_pd_mat(dim)

    with jax.disable_jit():
        distance1 = dist.frob_gevd_weighted_fullrank(A, B)
        distance2 = dist.frob_gevd_weighted(A, B, rank = dim)

    assert np.allclose(distance1, distance2)