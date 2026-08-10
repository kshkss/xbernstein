from typing import ClassVar

from ._tensor_bernstein import _TensorBernstein


class Bernstein4D(_TensorBernstein):
    r"""Represent a tensor-product Bernstein polynomial on $[0,1]^4$.

    With coefficient array shape
    ``(*batch, n_0 + 1, n_1 + 1, n_2 + 1, n_3 + 1)``, the final four axes
    store $c_{i_0,i_1,i_2,i_3}$ and the polynomial is

    $$
    p(\mathbf{u})=
    \sum_{i_0=0}^{n_0}\sum_{i_1=0}^{n_1}
    \sum_{i_2=0}^{n_2}\sum_{i_3=0}^{n_3}
    c_{i_0,i_1,i_2,i_3}
    \prod_{a=0}^{3}B_{i_a}^{n_a}(u_a).
    $$
    """

    parameter_dimensions: ClassVar[int] = 4
