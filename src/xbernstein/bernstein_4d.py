from typing import ClassVar

from ._tensor_bernstein import _TensorBernstein


class Bernstein4D(_TensorBernstein):
    """Tensor-product Bernstein polynomial on the unit hypercube."""

    parameter_dimensions: ClassVar[int] = 4
