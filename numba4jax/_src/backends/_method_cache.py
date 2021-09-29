from ..config_flags import config

_DEBUG = config.FLAGS["NUMBA4JAX_DEBUG"]

__method_cache = {}


def _get_backend_cache(backend):
    if not backend in __method_cache:
        if _DEBUG:
            print(f"Initializing method cache for backend {backend}.")

        __method_cache[backend] = {}

    return __method_cache[backend]


def _get_function_cache(backend, fun):
    __backend_cache = _get_backend_cache(backend)

    if not fun in __backend_cache:
        if _DEBUG:
            print(f"Initializing method cache ({backend}) for function {fun}.")

        __backend_cache[fun] = {}

    return __backend_cache[fun]


def get_custom_call_name(
    backend,
    numba_fn,
    *,
    input_shapes,
    input_dtypes,
    output_shapes,
    output_dtypes,
    compile_fun,
):
    _method_cache = _get_function_cache(backend, numba_fn)

    # could make shapes dynamic in C and so cache on them.
    # Not sure this is useful.

    # input_ndims = tuple(len(shape) for shape in input_shapes)
    # output_ndims = tuple(len(shape) for shape in input_shapes)
    # key = (input_ndims, input_dtypes, output_ndims, output_dtypes)

    key = (input_shapes, input_dtypes, output_shapes, output_dtypes)

    if not key in _method_cache:
        if _DEBUG:
            print(f"Method ({numba_fn})[{key}] not found in cache. Compiling...")

        _method_cache[key] = compile_fun(
            numba_fn,
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
        )

    return _method_cache[key]
