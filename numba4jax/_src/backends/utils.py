from jax.interpreters.mlir import token_type

import jaxlib.mlir.ir as ir


def get_default_layouts(operands, order="c"):
    token = token_type()
    layouts = []

    if order == "c":
        default_layout = lambda t: tuple(range(len(t.shape) - 1, -1, -1))  # noqa: E731
    elif order == "f":
        default_layout = lambda t: tuple(range(len(t.shape)))  # noqa: E731
    else:
        raise ValueError(f"Unknown order: {order}")

    for op in operands:
        if isinstance(op, (ir.Value)):
            if op.type == token:
                layouts.append(())
            else:
                tensor_type = ir.RankedTensorType(op.type)
                layouts.append(default_layout(tensor_type))

        elif isinstance(op, ir.RankedTensorType):
            layouts.append(default_layout(op))

        elif op == token:
            layouts.append(())

        else:
            raise ValueError(f"Unknown operand type: {type(op)}")

    return layouts
