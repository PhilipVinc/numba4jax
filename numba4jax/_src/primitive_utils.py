import collections
import ctypes
from functools import partial  # pylint:disable=g-importing-member
from textwrap import dedent

import jax
import jax.numpy as jnp
from jax.interpreters import batching
from jax.interpreters import xla
from jax.lib import xla_client

import numba
from numba import types as nb_types
import numpy as np

from .config_flags import config
from . import xla_utils
from . import backends


def abstract_eval_rule(abstract_eval_fn, *args, **kwargs):
    """
    Evaluates abstract_eval_fn and returns an iterable of
    the results.

    This function makes sure that the returned object is always
    a PyTree, even if the function only has 1 result.
    """
    # Special-casing when only a single tensor is returned.
    shapes = abstract_eval_fn(*args, **kwargs)
    if not isinstance(shapes, collections.abc.Collection):
        return [shapes]
    else:
        return shapes


def bind_primitive(primitive, abstract_eval_fn, *args):
    """
    Binds a primitive and returns a pytree only if the result
    is a tuple of 2 or more onbjects.
    """
    result = primitive.bind(*args)

    output_shapes = abstract_eval_fn(*args)
    # Special-casing when only a single tensor is returned.
    if not isinstance(output_shapes, collections.abc.Collection):
        assert len(result) == 1
        return result[0]
    else:
        return result


def eval_rule(call_fn, abstract_eval_fn, *args, **kwargs):
    """
    Python Evaluation rule for a numba4jax function respecting the
    XLA CustomCall interface.

    Evaluates `outs = abstract_eval_fn(*args)` to compute the output shape
    and preallocate them, then executes `call_fn(*outs, *args)` which is
    the Numba kernel.

    Args:
        call_fn: a (numba.jit) function respecting the calling convention of
            XLA CustomCall, taking first the outputs by reference then the
            inputs.
        abstract_eval_fn: The abstract evaluation function respecting jax
            interface
        args: The arguments to the `call_fn`
        kwargs: Optional keyword arguments for the numba function.
    """

    # compute the output shapes
    output_shapes = abstract_eval_fn(*args)
    # Preallocate the outputs
    outputs = tuple(np.empty(shape.shape, dtype=shape.dtype) for shape in output_shapes)
    # convert inputs to a tuple
    inputs = tuple(np.asarray(arg) for arg in args)
    # call the kernel
    call_fn(outputs + inputs, **kwargs)
    # Return the outputs
    return tuple(outputs)


def naive_batching_rule(call_fn, args, batch_axes):
    """
    Returns the batching rule for a numba4jax kernel, which simply
    maps the call among the batched axes.
    """
    # TODO(josipd): Check that the axes are all zeros. Add support when only a
    #               subset of the arguments have to be batched.
    # TODO(josipd): Do this smarter than n CustomCalls.
    print(f"batching {call_fn} with {args} and {batch_axes}")
    result = jax.lax.map(lambda x: call_fn(*x), args)
    print(f"batching gives result {result.shape} over axis {batch_axes}")
    print(
        "result has shape:",
    )
    for p in result:
        print("  ", p.shape)

    return result, batch_axes


def numba_to_jax(name: str, numba_fn, abstract_eval_fn, batching_fn=None):
    """Create a jittable JAX function for the given Numba function.

    Args:
      name: The name under which the primitive will be registered.
      numba_fn: The function that can be compiled with Numba.
      abstract_eval_fn: The abstract evaluation function.
      batching_fn: If set, this function will be used when vmap-ing the returned
        function.
    Returns:
      A jitable JAX function.
    """
    primitive = jax.core.Primitive(name)
    primitive.multiple_results = True

    abstract_eval = partial(abstract_eval_rule, abstract_eval_fn)
    bind_primitive_fn = partial(bind_primitive, primitive, abstract_eval_fn)

    primitive.def_abstract_eval(abstract_eval)
    primitive.def_impl(partial(eval_rule, numba_fn, abstract_eval))

    # if batching_fn is not None:
    #    batching.primitive_batchers[primitive] = batching_fn
    # else:
    #    batching.primitive_batchers[primitive] = partial(
    #        naive_batching_rule, bind_primitive_fn
    #    )
    # batching.defvectorized(primitive)

    xla.backend_specific_translations["cpu"][primitive] = partial(
        backends.cpu.xla_encode, numba_fn, abstract_eval
    )

    xla.backend_specific_translations["gpu"][primitive] = partial(
        backends.gpu.xla_encode, numba_fn, abstract_eval
    )

    return bind_primitive_fn


def njit4jax(output_shapes):
    """Function decorator equivalent to `numba.jit(nopython=True)` and then converting
    the resulting numba-jitted function in a Jax/XLA compatible primitive.

    Args:
        output_shapes: The shape of the resulting function (type-annotation
            of the output). This is necessary because jax/python can deduce the types of
            inputs when you call it, but it cannot infer the types of the output.
            `output_shapes` can be a function or a PyTree. If it is a pytree, it is
            assumed that the function always returns objects of the same type, regardless
            of the input types/shapes. If it is a function, it takes as input the
            argument shapes and dtypes and should return the pytree of correct output
            shapes of `jax.abstract_arrays.ShapedArray`.

    Example:
        > from numba4jax import ShapedArray, njit4jax
        > @njit4jax(lambda x: ShapedArray)
        > def myadd(x):
        >
    """

    # If output_shapes is callable use it as abstract evaluation,
    # otherwise assume constant shape output.
    if callable(output_shapes):
        abstract_eval = output_shapes
    else:
        abstract_eval = lambda *args: output_shapes

    def decorator(fun):
        jitted_fun = numba.njit(fun)
        fn_name = "{}::{}".format(
            xla_utils.default_primitive_name(fun), hash(jitted_fun)
        )

        if config.FLAGS["NUMBA4JAX_DEBUG"]:
            print("Constructing CustomCall function numbaa4jax:", fn_name)

        return numba_to_jax(fn_name, jitted_fun, abstract_eval)

    return decorator
