import numba
from numba import types as nb_types

import numpy as np

from jax.interpreters import mlir
from jax.interpreters.mlir import custom_call  # noqa: F401
from jax.lib import xla_client

import jaxlib.mlir.ir as ir

from numba4jax._src import xla_utils
from numba4jax._src import config

from . import _cuda as cuda
from .utils import get_default_layouts
from ._method_cache import get_custom_call_name

xla_call_sig = nb_types.void(
    nb_types.voidptr,  # cudaStream_t* stream
    nb_types.CPointer(nb_types.voidptr),  # void** buffers
    nb_types.voidptr,  # const char* opaque
    nb_types.uint64,  # size_t opaque_len
)


def compile_gpu_signature(
    numba_fn, *, input_shapes, input_dtypes, output_shapes, output_dtypes
):
    """
    Compiles numba_fn to C and register it with XLA for the given signature.
    """
    from ._cuda import (
        cuMemcpyAsync,
        cuStreamSynchronize,
        memcpyHostToDevice,
        memcpyDeviceToHost,
    )

    n_in = len(input_shapes)
    n_out = len(output_shapes)

    input_byte_size = tuple(
        np.prod(shape) * dtype.itemsize
        for (shape, dtype) in zip(input_shapes, input_dtypes)
    )
    output_byte_size = tuple(
        np.prod(shape) * dtype.itemsize
        for (shape, dtype) in zip(output_shapes, output_dtypes)
    )

    @numba.cfunc(xla_call_sig)
    def xla_custom_call_target(stream, inout_gpu_ptrs, opaque, opaque_len):
        # manually unroll input and output args because numba is
        # relatively dummb and cannot always infer getitem on inhomogeneous tuples

        # allocate output cpu bufferess
        if n_out == 1:
            args_out = (np.empty(output_shapes[0], dtype=output_dtypes[0]),)
        elif n_out == 2:
            args_out = (
                np.empty(output_shapes[0], dtype=output_dtypes[0]),
                np.empty(output_shapes[1], dtype=output_dtypes[1]),
            )
        elif n_out == 3:
            args_out = (
                np.empty(output_shapes[0], dtype=output_dtypes[0]),
                np.empty(output_shapes[1], dtype=output_dtypes[1]),
                np.empty(output_shapes[2], dtype=output_dtypes[2]),
            )
        elif n_out == 4:
            args_out = (
                np.empty(output_shapes[0], dtype=output_dtypes[0]),
                np.empty(output_shapes[1], dtype=output_dtypes[1]),
                np.empty(output_shapes[2], dtype=output_dtypes[2]),
                np.empty(output_shapes[3], dtype=output_dtypes[3]),
            )

        # allocate input cpu buffers and
        if n_in == 1:
            args_in = (np.empty(input_shapes[0], dtype=input_dtypes[0]),)
            cuMemcpyAsync(
                args_in[0].ctypes.data,
                inout_gpu_ptrs[0],
                input_byte_size[0],
                memcpyDeviceToHost,
                stream,
            )
        elif n_in == 2:
            args_in = (
                np.empty(input_shapes[0], dtype=input_dtypes[0]),
                np.empty(input_shapes[1], dtype=input_dtypes[1]),
            )
            cuMemcpyAsync(
                args_in[0].ctypes.data,
                inout_gpu_ptrs[0],
                input_byte_size[0],
                memcpyDeviceToHost,
                stream,
            )
            cuMemcpyAsync(
                args_in[1].ctypes.data,
                inout_gpu_ptrs[1],
                input_byte_size[1],
                memcpyDeviceToHost,
                stream,
            )
        elif n_in == 3:
            args_in = (
                np.empty(input_shapes[0], dtype=input_dtypes[0]),
                np.empty(input_shapes[1], dtype=input_dtypes[1]),
                np.empty(input_shapes[2], dtype=input_dtypes[2]),
            )
            cuMemcpyAsync(
                args_in[0].ctypes.data,
                inout_gpu_ptrs[0],
                input_byte_size[0],
                memcpyDeviceToHost,
                stream,
            )
            cuMemcpyAsync(
                args_in[1].ctypes.data,
                inout_gpu_ptrs[1],
                input_byte_size[1],
                memcpyDeviceToHost,
                stream,
            )
            cuMemcpyAsync(
                args_in[2].ctypes.data,
                inout_gpu_ptrs[2],
                input_byte_size[2],
                memcpyDeviceToHost,
                stream,
            )
        cuStreamSynchronize(stream)
        numba_fn(args_out + args_in)

        if n_out == 1:
            cuMemcpyAsync(
                inout_gpu_ptrs[n_in + 0],
                args_out[0].ctypes.data,
                output_byte_size[0],
                memcpyHostToDevice,
                stream,
            )
        elif n_out == 2:
            cuMemcpyAsync(
                inout_gpu_ptrs[n_in + 0],
                args_out[0].ctypes.data,
                output_byte_size[0],
                memcpyHostToDevice,
                stream,
            )
            cuMemcpyAsync(
                inout_gpu_ptrs[n_in + 1],
                args_out[1].ctypes.data,
                output_byte_size[1],
                memcpyHostToDevice,
                stream,
            )
        elif n_out == 3:
            cuMemcpyAsync(
                inout_gpu_ptrs[n_in + 0],
                args_out[0].ctypes.data,
                output_byte_size[0],
                memcpyHostToDevice,
                stream,
            )
            cuMemcpyAsync(
                inout_gpu_ptrs[n_in + 1],
                args_out[1].ctypes.data,
                output_byte_size[1],
                memcpyHostToDevice,
                stream,
            )
            cuMemcpyAsync(
                inout_gpu_ptrs[n_in + 2],
                args_out[2].ctypes.data,
                output_byte_size[2],
                memcpyHostToDevice,
                stream,
            )
        elif n_out == 4:
            cuMemcpyAsync(
                inout_gpu_ptrs[n_in + 0],
                args_out[0].ctypes.data,
                output_byte_size[0],
                memcpyHostToDevice,
                stream,
            )
            cuMemcpyAsync(
                inout_gpu_ptrs[n_in + 1],
                args_out[1].ctypes.data,
                output_byte_size[1],
                memcpyHostToDevice,
                stream,
            )
            cuMemcpyAsync(
                inout_gpu_ptrs[n_in + 2],
                args_out[2].ctypes.data,
                output_byte_size[2],
                memcpyHostToDevice,
                stream,
            )
            cuMemcpyAsync(
                inout_gpu_ptrs[n_in + 3],
                args_out[3].ctypes.data,
                output_byte_size[3],
                memcpyHostToDevice,
                stream,
            )

        cuStreamSynchronize(stream)

    target_name = xla_custom_call_target.native_name.encode("ascii")

    if config.FLAGS["NUMBA4JAX_DEBUG"]:
        print("primitive CustomCall rule name (xla_c_rule.native_name): ", target_name)

    # Extract the pointer to the CFFI function and create a pycapsule
    # around it
    capsule = xla_utils.create_xla_target_capsule(xla_custom_call_target.address)

    xla_client.register_custom_call_target(target_name, capsule, "gpu")

    return target_name


def xla_encode(numba_fn, abstract_eval_fn, ctx, *args):
    if not cuda.numba_cffi_loaded:
        raise RuntimeError("Numba cffi could not be loaded.")

    if config.FLAGS["NUMBA4JAX_DEBUG"]:
        print("Encoding the GPU variant of numba4jax function")

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
        "gpu",
        numba_fn,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        compile_fun=compile_gpu_signature,
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
