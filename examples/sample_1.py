import jax
import jax.numpy as jnp

from numba4jax import ShapedArray, njit4jax


def compute_type(*x):
    return x[0]


@njit4jax(compute_type)
def test(args):
    y, x, x2 = args
    y[:] = x[:] + 1


z = jnp.ones((1, 2), dtype=float)

jax.make_jaxpr(test)(z, z)

print("output: ", test(z, z))
print("output: ", jax.jit(test)(z, z))

z = jnp.ones((2, 3), dtype=float)
print("output: ", jax.jit(test)(z, z))

z = jnp.ones((1, 3, 1), dtype=float)
print("output: ", jax.jit(test)(z, z))
