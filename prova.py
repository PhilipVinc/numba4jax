import jax
import numpy as np
import jax.numpy as jnp
from jax import random

from numba4jax import njit4jax


@njit4jax(
    lambda x: jax.abstract_arrays.ShapedArray(x.shape, x.dtype),
)
def _transition(args):
    # unpack arguments
    xout, xin = args

    xout[:] = xin + 1


def prova(x):
    return _transition(x)


key = random.PRNGKey(3)
a = random.normal(key, (3,))

print(prova(a))

print(jax.jit(prova)(a))

print(jax.make_jaxpr(prova)(a))
print("--------")

vprova = jax.vmap(prova)
a = random.normal(key, (3, 4))

print(vprova(a))

print(jax.jit(vprova)(a))

print(jax.make_jaxpr(vprova)(a))
