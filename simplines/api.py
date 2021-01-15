from .linalg import StencilMatrix
from .spaces import TensorSpace

__all__ = ['assemble_matrix', 'assemble_vector']

#==============================================================================
def assemble_matrix(core, V, out=None):
    if out is None:
        out = StencilMatrix(V.vector_space, V.vector_space)

    # ...
    args = []
    if isinstance(V, TensorSpace):
        args += list(V.nelements)
        args += list(V.degree)
        args += list(V.spans)
        args += list(V.basis)
        args += list(V.weights)
        args += list(V.points)

    else:
        args = [V.nelements,
                V.degree,
                V.spans,
                V.basis,
                V.weights,
                V.points]
    # ...

    core( *args, out._data )

    return out

#==============================================================================
def assemble_vector(core, V, out=None):
    if out is None:
        out = StencilVector(V.vector_space)

    # ...
    args = []
    if isinstance(V, TensorSpace):
        args += list(V.nelements)
        args += list(V.degree)
        args += list(V.spans)
        args += list(V.basis)
        args += list(V.weights)
        args += list(V.points)

    else:
        args = [V.nelements,
                V.degree,
                V.spans,
                V.basis,
                V.weights,
                V.points]
    # ...

    core( *args, out._data )

    return out

