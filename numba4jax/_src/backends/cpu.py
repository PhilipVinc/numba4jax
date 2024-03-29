import numba
from numba import types as nb_types

from jax.interpreters import mlir
from jax.interpreters.mlir import custom_call  # noqa: F401
from jax.lib import xla_client

import jaxlib.mlir.ir as ir

from numba4jax._src import config
from numba4jax._src import xla_utils

from .utils import get_default_layouts
from ._method_cache import get_custom_call_name


def get_xla_call_signature(n_out):
    """
    See https://github.com/dfm/extending-jax

    It is documented that if there is a single output, it is directly passed as a single
    pointer, while if there are multiple outputs, we get out a tuple of pointers.
    """
    if n_out == 1:
        xla_call_sig = nb_types.void(
            nb_types.voidptr,  # output_ptrs
            nb_types.CPointer(nb_types.voidptr),  # input_ptrs
        )
    else:
        xla_call_sig = nb_types.void(
            nb_types.CPointer(nb_types.voidptr),  # output_ptrs
            nb_types.CPointer(nb_types.voidptr),  # input_ptrs
        )
    return xla_call_sig


def create_numba_api_wrapper(
    numba_fn, *, input_shapes, input_dtypes, output_shapes, output_dtypes
):
    """
    wraps `numba_fn` into another numba function respecting the XLA CustomCAll
    signature.
    This function is then exposed via the CFFI and its pointer returned.

    All kwargs are global constants for the numba function
    """

    n_in = len(input_shapes)
    n_out = len(output_shapes)

    if n_in > 4:
        raise NotImplementedError(
            "n_in ∈ [0,4] inputs are supported ({n_in} detected)."
            "Please open a bug report."
        )
    if n_out > 4 or n_out == 0:
        raise NotImplementedError(
            "n_out ∈ [1,4] outputs are supported ({n_out} detected)."
            "Please open a bug report."
        )
    xla_call_signature = get_xla_call_signature(n_out)

    @numba.cfunc(xla_call_signature)
    def xla_cpu_custom_call_target(output_ptrs, input_ptrs):
        # manually unroll input and output args because numba is
        # relatively dummb and cannot always infer getitem on inhomogeneous tuples
        if n_out == 1:
            args_out = (
                numba.carray(output_ptrs, output_shapes[0], dtype=output_dtypes[0]),
            )
        elif n_out == 2:
            args_out = (
                numba.carray(output_ptrs[0], output_shapes[0], dtype=output_dtypes[0]),
                numba.carray(output_ptrs[1], output_shapes[1], dtype=output_dtypes[1]),
            )
        elif n_out == 3:
            args_out = (
                numba.carray(output_ptrs[0], output_shapes[0], dtype=output_dtypes[0]),
                numba.carray(output_ptrs[1], output_shapes[1], dtype=output_dtypes[1]),
                numba.carray(output_ptrs[2], output_shapes[2], dtype=output_dtypes[2]),
            )
        elif n_out == 4:
            args_out = (
                numba.carray(output_ptrs[0], output_shapes[0], dtype=output_dtypes[0]),
                numba.carray(output_ptrs[1], output_shapes[1], dtype=output_dtypes[1]),
                numba.carray(output_ptrs[2], output_shapes[2], dtype=output_dtypes[2]),
                numba.carray(output_ptrs[3], output_shapes[3], dtype=output_dtypes[3]),
            )

        if n_in == 0:
            args_in = ()
        if n_in == 1:
            args_in = (
                numba.carray(input_ptrs[0], input_shapes[0], dtype=input_dtypes[0]),
            )
        elif n_in == 2:
            args_in = (
                numba.carray(input_ptrs[0], input_shapes[0], dtype=input_dtypes[0]),
                numba.carray(input_ptrs[1], input_shapes[1], dtype=input_dtypes[1]),
            )
        elif n_in == 3:
            args_in = (
                numba.carray(input_ptrs[0], input_shapes[0], dtype=input_dtypes[0]),
                numba.carray(input_ptrs[1], input_shapes[1], dtype=input_dtypes[1]),
                numba.carray(input_ptrs[2], input_shapes[2], dtype=input_dtypes[2]),
            )
        elif n_in == 4:
            args_in = (
                numba.carray(input_ptrs[0], input_shapes[0], dtype=input_dtypes[0]),
                numba.carray(input_ptrs[1], input_shapes[1], dtype=input_dtypes[1]),
                numba.carray(input_ptrs[2], input_shapes[2], dtype=input_dtypes[2]),
                numba.carray(input_ptrs[3], input_shapes[3], dtype=input_dtypes[3]),
            )

        numba_fn(args_out + args_in)

    return xla_cpu_custom_call_target


def compile_cpu_signature(
    numba_fn, *, input_shapes, input_dtypes, output_shapes, output_dtypes
):
    """
    Compiles numba_fn to C and register it with XLA for the given signature.
    """
    xla_c_rule = create_numba_api_wrapper(
        numba_fn,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )

    target_name = xla_c_rule.native_name.encode("ascii")

    if config.FLAGS["NUMBA4JAX_DEBUG"]:
        print("primitive CustomCall rule name (xla_c_rule.native_name): ", target_name)

    # Extract the pointer to the CFFI function and create a pycapsule
    # around it
    capsule = xla_utils.create_xla_target_capsule(xla_c_rule.address)

    xla_client.register_custom_call_target(target_name, capsule, "cpu")

    return target_name


def xla_encode(numba_fn, abstract_eval_fn, ctx, *args):
    """Returns the XLA CustomCall for the given numba function.

    Args:
      numba_fn: A numba function. For its signature, see the module docstring.
      abstract_eval_fn: The abstract shape evaluation function.
      xla_builder: The XlaBuilder instance.
      *args: The positional arguments to be passed to `numba_fn`.
    Returns:
      The XLA CustomCall operation calling into the numba function.
    """

    if config.FLAGS["NUMBA4JAX_DEBUG"]:
        print("Encoding the CPU variant of numba4jax function")

    inputs_avals = ctx.avals_in
    input_shapes = tuple(arg.shape for arg in inputs_avals)
    input_dtypes = tuple(arg.dtype for arg in inputs_avals)

    # TODO(josipd): Check that the input layout is the numpy default.
    output_abstract_arrays = abstract_eval_fn(*inputs_avals)

    output_shapes = tuple(array.shape for array in output_abstract_arrays)
    output_dtypes = tuple(array.dtype for array in output_abstract_arrays)
    output_types = [
        ir.RankedTensorType.get(o.shape, mlir.dtype_to_ir_type(o.dtype))
        for o in output_abstract_arrays
    ]

    target_name = get_custom_call_name(
        "cpu",
        numba_fn,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        compile_fun=compile_cpu_signature,
    )

    return custom_call(
        target_name,
        result_types=output_types,
        operands=args,
        # layout matters here, because the first axis is special
        operand_layouts=get_default_layouts(args),
        result_layouts=get_default_layouts(output_types),
        has_side_effect=False,
    ).results
