from typing import ClassVar

import jax
from jaxtyping import Float

from ._tensor_bernstein import _TensorBernstein, _tensor_minimize, _tensor_minimize_jvp


@jax.custom_jvp
def _minimize(
    coefficients: Float[jax.Array, "..."], max_steps: int = 200, eps: float = 1e-6
) -> tuple[Float[jax.Array, ""], Float[jax.Array, "dim"]]:
    return _tensor_minimize(coefficients, max_steps, eps)


@_minimize.defjvp
def _minimize_jvp(
    primals: tuple[Float[jax.Array, "..."], int, float],
    tangents: tuple[Float[jax.Array, "..."], int, float],
) -> tuple[tuple[Float[jax.Array, ""], Float[jax.Array, "dim"]], tuple[Float[jax.Array, ""], Float[jax.Array, "dim"]]]:
    coefficients, max_steps, eps = primals
    tangent_coefficients, _, _ = tangents
    return _tensor_minimize_jvp(coefficients, tangent_coefficients, max_steps, eps)


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

    Batch dimensions
    ----------------

    For coefficients of shape ``(2, 2, 3, 2, 2)``, the leading axis batches
    two polynomials and the trailing axes have degree vector $(1,2,1,1)$:

    ```python
    >>> import jax.numpy as jnp
    >>> from xbernstein import Bernstein4D
    >>> hypervolumes = Bernstein4D(jnp.zeros((2, 2, 3, 2, 2)))
    >>> hypervolumes(jnp.linspace(0.0, 1.0, 4), 0.5, 0.25, 0.75).shape
    (2, 4)
    >>> hypervolumes(jnp.zeros((3, 1)), jnp.zeros((1, 4)), 0.25, 0.75).shape
    (2, 3, 4)
    >>> left = Bernstein4D(jnp.zeros((2, 1, 2, 3, 2, 2)))
    >>> right = Bernstein4D(jnp.zeros((1, 3, 3, 2, 2, 2)))
    >>> (left + right).c.shape
    (2, 3, 3, 3, 2, 2)
    >>> try:
    ...     hypervolumes + Bernstein4D(jnp.zeros((3, 2, 3, 2, 2)))
    ... except ValueError as error:
    ...     type(error).__name__
    'ValueError'
    ```
    """

    parameter_dimensions: ClassVar[int] = 4
