from typing import ClassVar

from ._tensor_bernstein import _TensorBernstein


class Bernstein3D(_TensorBernstein):
    """Tensor-product Bernstein polynomial on the unit cube."""

    parameter_dimensions: ClassVar[int] = 3
