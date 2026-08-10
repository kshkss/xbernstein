from typing import ClassVar

from ._tensor_bernstein import _TensorBernstein


class Bernstein2D(_TensorBernstein):
    r"""Represent a tensor-product Bernstein polynomial on $[0,1]^2$.

    With coefficient array shape ``(*batch, n_0 + 1, n_1 + 1)``, the final
    two axes store $c_{i_0,i_1}$ and the polynomial is

    $$
    p(u_0,u_1)=
    \sum_{i_0=0}^{n_0}\sum_{i_1=0}^{n_1}
    c_{i_0,i_1}B_{i_0}^{n_0}(u_0)B_{i_1}^{n_1}(u_1).
    $$
    """

    parameter_dimensions: ClassVar[int] = 2
