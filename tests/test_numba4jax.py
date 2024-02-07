import jax
import jax.numpy as jnp
import numpy as np

from numba4jax import njit4jax


def test_onereturn():
    def compute_type(*x):
        return x[0]

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


def test_tworeturn():
    def compute_type(*x):
        return x[0], x[1]

    @njit4jax(compute_type)
    def test(args):
        y, y2, x, x2 = args
        y[:] = x[:] + 1
        y2[:] = x2[:] + 1

    z1 = jnp.ones((1, 2), dtype=float)
    z2 = jnp.ones((1, 2), dtype=float) * 10
    out1, out2 = test(z1, z2)
    out1_jit, out2_jit = jax.jit(test)(z1, z2)
    np.testing.assert_allclose(out1, out1_jit)
    np.testing.assert_allclose(out2, out2_jit)

    z1 = jnp.ones((2, 3, 2), dtype=float)
    z2 = jnp.ones((2, 3, 2), dtype=float) * 10
    out1, out2 = test(z1, z2)
    out1_jit, out2_jit = jax.jit(test)(z1, z2)
    np.testing.assert_allclose(out1, out1_jit)
    np.testing.assert_allclose(out2, out2_jit)
