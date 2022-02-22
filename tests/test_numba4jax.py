import jax
import jax.numpy as jnp
import numpy as np

from numba4jax import njit4jax


def compute_type(*x):
    return x[0]


def test_main():
    @njit4jax(compute_type)
    def test(args):
        y, x, x2 = args
        y[:] = x[:] + 1

    z = jnp.ones((1, 2), dtype=float)
    out = test(z, z)
    out_jit = jax.jit(test)(z, z)
    np.testing.assert_allclose(out, out_jit)

    z = jnp.ones((2, 3), dtype=float)
    out = test(z, z)
    out_jit = jax.jit(test)(z, z)
    np.testing.assert_allclose(out, out_jit)

    z = jnp.ones((1, 3, 1), dtype=float)
    out = test(z, z)
    out_jit = jax.jit(test)(z, z)
    np.testing.assert_allclose(out, out_jit)
