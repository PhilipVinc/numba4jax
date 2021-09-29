from numba import cuda as ncuda
from cffi import FFI

from numba import types as nb_types

from numba4jax._src.c_api import find_path_of_symbol_in_library

try:
    _libcuda = ncuda.driver.find_driver()

    libcuda_path = find_path_of_symbol_in_library(_libcuda.cuMemcpy)

    numba_cffi_loaded = True
except:
    numba_cffi_loaded = False

if numba_cffi_loaded:

    # functions needed
    ffi = FFI()
    ffi.cdef("int cuMemcpy(void* dst, void* src, unsigned int len, int type);")
    ffi.cdef(
        "int cuMemcpyAsync(void* dst, void* src, unsigned int len, int type, void* stream);"
    )
    ffi.cdef("int cuStreamSynchronize(void* stream);")

    ffi.cdef("int cudaMallocHost(void** ptr, size_t size);")
    ffi.cdef("int cudaFreeHost(void* ptr);")

    # load libraray
    # could  ncuda.driver.find_library()
    libcuda = ffi.dlopen(libcuda_path)
    cuMemcpy = libcuda.cuMemcpy
    cuMemcpyAsync = libcuda.cuMemcpyAsync
    cuStreamSynchronize = libcuda.cuStreamSynchronize

    memcpyHostToHost = nb_types.int32(0)
    memcpyHostToDevice = nb_types.int32(1)
    memcpyDeviceToHost = nb_types.int32(2)
    memcpyDeviceToDevice = nb_types.int32(3)
