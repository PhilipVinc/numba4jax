import sys

import ctypes
import ctypes.util

from ctypes import pythonapi

# The code below defines the calling signature
# of some Python C-api functions that we need to
# call because they are not exposed as native
# Python methods.

# Declaring the signature requires setting
# the `argtypes` and `restype` of a valid
# symbol defined in the pythonlib.DLL

# Define `PyCapsule_New`
pythonapi.PyCapsule_New.argtypes = [
    ctypes.c_void_p,  # void* pointer
    ctypes.c_char_p,  # const char *name
    ctypes.c_void_p,  # PyCapsule_Destructor destructor
]
pythonapi.PyCapsule_New.restype = ctypes.py_object


def pycapsule_new(ptr, name, destructor=None) -> ctypes.py_object:
    """
    Wraps a C function pointer into an XLA-compatible PyCapsule.

    Args:
        ptr: A CFFI pointer to a function
        name: A binary string
        destructor: Optional PyCapsule object run at destruction

    Returns
        a PyCapsule (ctypes.py_object)
    """
    return ctypes.pythonapi.PyCapsule_New(ptr, name, None)


# Find the dynamic linker library path
libdl_path = ctypes.util.find_library("dl")
# Load the dynamic linker dynamically
libdl = ctypes.CDLL(libdl_path)


class Dl_info(ctypes.Structure):
    """
    Structure of the Dl_info returned by the CFFI of dl.dladdr
    """

    _fields_ = (
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    )


# Define dladdr to get the pointer to a symbol in a shared
# library already loaded.
# https://man7.org/linux/man-pages/man3/dladdr.3.html
libdl.dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(Dl_info))
# restype is None as it returns by reference


def find_path_of_symbol_in_library(symbol):
    info = Dl_info()

    result = libdl.dladdr(symbol, ctypes.byref(info))

    if result and info.dli_fname:
        return info.dli_fname.decode(sys.getfilesystemencoding())
    else:
        raise ValueError("Cannot determine path of Library.")
        libdl_path = "Not Found"
