from typing import ClassVar

from ._tensor_bernstein import _TensorBernstein


class Bernstein2D(_TensorBernstein):
    """Tensor-product Bernstein polynomial on the unit square."""

    parameter_dimensions: ClassVar[int] = 2
