from .bernstein import Bernstein, _minimize
from .bernstein_2d import Bernstein2D
from .bernstein_3d import Bernstein3D
from .bernstein_4d import Bernstein4D

__all__ = ["Bernstein", "Bernstein2D", "Bernstein3D", "Bernstein4D", "minimize"]


import jax
from jaxtyping import Float
from typing import NamedTuple


class OptimizeResult(NamedTuple):
    r"""Store an approximation to $\min_{t\in[0,1]}p(t)$ and an associated $t$.

    ``f`` is the current upper-bound value and ``x`` is the parameter at which
    that value is attained. Both arrays have the same leading batch/value
    shape as the input polynomial.

    Batch dimensions
    ----------------

    If the input coefficient array has shape ``(*batch, n + 1)``, both ``f``
    and ``x`` have shape ``(*batch,)``. Each leading index is minimized
    independently.

    ```python
    >>> import jax.numpy as jnp
    >>> result = OptimizeResult(f=jnp.zeros((2, 3)), x=jnp.ones((2, 3)))
    >>> result.f.shape, result.x.shape
    ((2, 3), (2, 3))
    ```
    """

    f: Float[jax.Array, "*batch"]  # 最小値
    x: Float[jax.Array, "*batch"]  # 最小値を与えるx

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the leading shape shared by the minimum value $f$ and point $x$."""
        return self.f.shape

    @property
    def dtype(self) -> str:
        """Return the scalar field used for $f$ and $x$."""
        return str(self.f.dtype)


def minimize(
    bpoly: Float[Bernstein, "*batch"], max_steps: int = 200, eps: float = 1e-6
) -> Float[OptimizeResult, "*batch"]:
    r"""Apply branch-and-bound to each leading-axis polynomial on $[0,1]$.

    The method computes an approximate pair

    $$
    (f,x)\approx\left(\min_{t\in[0,1]}p(t),
    \mathop{\mathrm{argmin}}_{t\in[0,1]}p(t)\right)
    $$

    for every independently batched coefficient vector. It flattens leading
    axes only to vectorize the scalar solver, then restores their original
    arrangement in :class:`OptimizeResult`.

    Batch dimensions
    ----------------

    The final coefficient axis is excluded from the batch shape. The following
    call minimizes two degree-1 polynomials independently:

    ```python
    >>> import jax.numpy as jnp
    >>> from xbernstein import Bernstein
    >>> from xbernstein import minimize
    >>> result = minimize(Bernstein(jnp.array([[0.0, 1.0], [1.0, 0.0]])))
    >>> result.f.shape, result.x.shape
    ((2,), (2,))
    ```

    A higher-rank leading shape is likewise preserved:

    ```python
    >>> result = minimize(Bernstein(jnp.zeros((2, 3, 2))))
    >>> result.f.shape
    (2, 3)
    ```

    ``minimize`` accepts one Bernstein object, so it does not broadcast
    separate polynomial inputs. The leading shape is taken entirely from that
    object's coefficient array.
    """
    shape = bpoly.shape
    n = bpoly.order + 1
    coefficients = bpoly.c.reshape([-1, n])

    fs, xs = jax.vmap(_minimize, in_axes=(0, None, None))(coefficients, max_steps, eps)
    results = OptimizeResult(f=fs.reshape(shape), x=xs.reshape(shape))
    return results
