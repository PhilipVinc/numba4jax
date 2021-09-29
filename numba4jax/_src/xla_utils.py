import jax

from jax.abstract_arrays import ShapedArray

from .c_api import pycapsule_new


def xla_shape_to_abstract(xla_shape) -> ShapedArray:
    """
    Converts an XLA shape to a Jax ShapedArray object, which
    is the empty shell defining only shape and dtype used by
    abstract evaluation
    """
    return ShapedArray(xla_shape.dimensions(), xla_shape.element_type())


def create_xla_target_capsule(ptr):
    """
    Wraps a C function pointer into an XLA-compatible PyCapsule.

    Assumes that the function pointed at by the pointer `ptr`
    respects the XLA calling convention (3 void* arguments).

    """
    # Magic name that the PyCapsule must have to be recognized
    # by XLA as a custom call
    xla_capsule_magic = b"xla._CUSTOM_CALL_TARGET"

    return pycapsule_new(ptr, xla_capsule_magic)


def default_primitive_name(fun) -> str:
    return f"njit4jax[{fun.__module__}.{fun.__name__}]"
