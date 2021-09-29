"""Call Numba from jitted JAX functions.

# The interface

To call your Numba function from JAX, you have to implement:

  1. A Numba function following our calling convention.
  2. A function for abstractly evaluating the function, i.e., for specifying
     the output shapes and dtypes from the input ones.

## 1. The Numba function

The Numba function has to accept a *single* tuple argument and do not return
anythin, i.e. have type `Callable[tuple[numba.carray], None]`. The output and
input arguments are stored consecutively in the tuple. For example, if you want
to implement a function that takes three arrays and returns two, the Numba
function should look like:

```py
@numba.jit
def add_and_mul(args):
  output_1, output_2, input_1, input_2, input_3 = args
  # Now edit output_1 and output_2 *in place*.
  output_1.fill(0)
  output_2.fill(0)
  output_1 += input_1 + input_2
  output_2 += input_1 * input_3
```

Note that the output arguments have to be modified *in-place*. These arrays are
allocated and owned by XLA.

## 2. The abstract evaluation function

You also have to implement a function that tells JAX how to compute the shapes
and types of the outputs from the inputs.For more information, please refer to

https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html#Abstract-evaluation-rules

For example, for the above function, the corresponding abstract eval function is

```py
def add_and_mul_shape_fn(input_1, input_2, input_3):
  assert input_1.shape == input_2.shape
  assert input_1.shape == input_3.shape
  return (jax.abstract_arrays.ShapedArray(input_1.shape, input_1.dtype),
          jax.abstract_arrays.ShapedArray(input_1.shape, input_1.dtype))
```

# Conversion

Now, what is left is to convert the function:

```py
add_and_mul_jax = jax.experimental.jambax.numba_to_jax(
    "add_and_mul", add_and_mul, add_and_mul_shape_fn)
```

You can JIT compile the function as
```py
add_and_mul_jit = jax.jit(add_and_mul_jax)
```

# Optional
## Derivatives

You can define a gradient for your function as if you were definining a custom
gradient for any other JAX function. You can follow the tutorial at:

https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html

## Batching / vmap

Batching along the first axes is implemented via jax.lax.map. To implement your
own bathing rule, see the documentation of `numba_to_jax`.
"""

from .config_flags import config

from .primitive_utils import njit4jax
