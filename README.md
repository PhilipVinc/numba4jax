# numba4jax



A small experimental python package allowing you to use numba-jitted functions from within jax 
with no overhead.

This package uses the CFFI of Numba to expose the C Function pointer of your compiled
function to XLA. It works both for CPU and GPU functions.

This package exports a single decorator `@njit4jax`, which takes an argument, a function
or Tuple describing the output shape of the function itself.
See the brief example below.

```python

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

```

## Backend support

This package supports both the CPU and GPU backends of jax.
The GPU backend is only supported on linux, and is highly experimental.
It requires CUDA to be installed in a standard path.
CUDA is found through `numba.cuda`, so you should first check that `numba.cuda`
works.
