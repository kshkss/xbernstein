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


class Bernstein2D(_TensorBernstein):
    r"""Represent a tensor-product Bernstein polynomial on $[0,1]^2$.

    With coefficient array shape ``(*batch, n_0 + 1, n_1 + 1)``, the final
    two axes store $c_{i_0,i_1}$ and the polynomial is

    $$
    p(u_0,u_1)=
    \sum_{i_0=0}^{n_0}\sum_{i_1=0}^{n_1}
    c_{i_0,i_1}B_{i_0}^{n_0}(u_0)B_{i_1}^{n_1}(u_1).
    $$

    Batch dimensions
    ----------------

    Every leading coefficient axis selects an independent patch. For example,
    the coefficient shape ``(2, 2, 3)`` represents two patches with degree
    vector $(1,2)$:

    ```python
    >>> import jax.numpy as jnp
    >>> from xbernstein import Bernstein2D
    >>> patches = Bernstein2D(jnp.zeros((2, 2, 3)))
    >>> patches(jnp.linspace(0.0, 1.0, 4), 0.5).shape
    (2, 4)
    >>> patches(jnp.zeros((3, 1)), jnp.zeros((1, 4))).shape
    (2, 3, 4)
    >>> left = Bernstein2D(jnp.zeros((2, 1, 2, 3)))
    >>> right = Bernstein2D(jnp.zeros((1, 3, 3, 2)))
    >>> (left + right).c.shape
    (2, 3, 3, 3)
    >>> try:
    ...     patches + Bernstein2D(jnp.zeros((3, 2, 3)))
    ... except ValueError as error:
    ...     type(error).__name__
    'ValueError'
    ```
    """

    parameter_dimensions: ClassVar[int] = 2
