from .bernstein import Bernstein, _minimize
from .bernstein_2d import Bernstein2D, _minimize as _minimize_2d
from .bernstein_3d import Bernstein3D, _minimize as _minimize_3d
from .bernstein_4d import Bernstein4D, _minimize as _minimize_4d
from .rational_bernstein import RationalBernstein, _minimize as _minimize_rational
from .rational_tensor_bernstein import (
    RationalBernstein2D,
    RationalBernstein3D,
    RationalBernstein4D,
    _minimize as _minimize_rational_tensor,
)
from .hermite import (
    hermite_interpolate_1d,
    hermite_interpolate_2d,
    hermite_interpolate_3d,
    hermite_interpolate_4d,
    linear_interpolate_1d,
    linear_interpolate_2d,
    linear_interpolate_3d,
    linear_interpolate_4d,
    quintic_hermite_interpolate_1d,
    quintic_hermite_interpolate_2d,
    quintic_hermite_interpolate_3d,
    quintic_hermite_interpolate_4d,
)

__all__ = [
    "Bernstein",
    "Bernstein2D",
    "Bernstein3D",
    "Bernstein4D",
    "RationalBernstein",
    "RationalBernstein2D",
    "RationalBernstein3D",
    "RationalBernstein4D",
    "linear_interpolate_1d",
    "linear_interpolate_2d",
    "linear_interpolate_3d",
    "linear_interpolate_4d",
    "hermite_interpolate_1d",
    "hermite_interpolate_2d",
    "hermite_interpolate_3d",
    "hermite_interpolate_4d",
    "quintic_hermite_interpolate_1d",
    "quintic_hermite_interpolate_2d",
    "quintic_hermite_interpolate_3d",
    "quintic_hermite_interpolate_4d",
    "maximize",
    "minimize",
]


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
    bpoly: (
        Bernstein
        | Bernstein2D
        | Bernstein3D
        | Bernstein4D
        | RationalBernstein
        | RationalBernstein2D
        | RationalBernstein3D
        | RationalBernstein4D
    ),
    max_steps: int = 200,
    eps: float = 1e-6,
) -> OptimizeResult:
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

    ``minimize`` accepts one polynomial object, so it does not broadcast
    separate inputs. The leading shape is taken entirely from that object's
    coefficient array. Positive-weight :class:`RationalBernstein` functions
    use the convex hull of their dehomogenized control values for bounds.
    """
    if isinstance(bpoly, RationalBernstein):
        shape = bpoly.shape
        homogeneous = bpoly.h.reshape((-1,) + bpoly.h.shape[-2:])
        fs, xs = jax.vmap(_minimize_rational, in_axes=(0, None, None))(
            homogeneous, max_steps, eps
        )
        return OptimizeResult(f=fs.reshape(shape), x=xs.reshape(shape))

    rational_tensor_types = (
        RationalBernstein2D,
        RationalBernstein3D,
        RationalBernstein4D,
    )
    for dimensions, rational_type in enumerate(rational_tensor_types, start=2):
        if isinstance(bpoly, rational_type):
            shape = bpoly.shape
            homogeneous_shape = bpoly.h.shape[-(dimensions + 1) :]
            homogeneous = bpoly.h.reshape((-1,) + homogeneous_shape)
            fs, xs = jax.vmap(
                _minimize_rational_tensor, in_axes=(0, None, None)
            )(homogeneous, max_steps, eps)
            return OptimizeResult(
                f=fs.reshape(shape),
                x=xs.reshape(shape + (dimensions,)),
            )

    solvers = (
        (Bernstein, _minimize, 1),
        (Bernstein2D, _minimize_2d, 2),
        (Bernstein3D, _minimize_3d, 3),
        (Bernstein4D, _minimize_4d, 4),
    )
    for polynomial_type, solver, dimensions in solvers:
        if isinstance(bpoly, polynomial_type):
            shape = bpoly.shape
            coefficient_shape = bpoly.c.shape[-dimensions:]
            coefficients = bpoly.c.reshape((-1,) + coefficient_shape)
            fs, xs = jax.vmap(solver, in_axes=(0, None, None))(
                coefficients, max_steps, eps
            )
            x_shape = shape if dimensions == 1 else shape + (dimensions,)
            return OptimizeResult(f=fs.reshape(shape), x=xs.reshape(x_shape))
    raise TypeError(
        f"minimize requires a Bernstein polynomial, got {type(bpoly).__name__}"
    )


def maximize(
    bpoly: (
        Bernstein
        | Bernstein2D
        | Bernstein3D
        | Bernstein4D
        | RationalBernstein
        | RationalBernstein2D
        | RationalBernstein3D
        | RationalBernstein4D
    ),
    max_steps: int = 200,
    eps: float = 1e-6,
) -> OptimizeResult:
    r"""Approximate the global maximum of a supported Bernstein representation.

    The implementation uses

    $$
    \max_{\mathbf{u}}p(\mathbf{u})
    =-\min_{\mathbf{u}}\left[-p(\mathbf{u})\right].
    $$

    It therefore has the same branch-and-bound convergence and batching
    behavior as :func:`minimize`. For a one-dimensional polynomial or rational
    function, ``f`` and ``x`` have shape ``(*batch,)``; for a $d$-dimensional
    tensor polynomial, they have shapes ``(*batch,)`` and ``(*batch, d)``,
    respectively.
    """
    rational_types = (
        RationalBernstein,
        RationalBernstein2D,
        RationalBernstein3D,
        RationalBernstein4D,
    )
    if isinstance(bpoly, RationalBernstein):
        negated = RationalBernstein(-bpoly.values, bpoly.weights)
    elif isinstance(bpoly, rational_types[1:]):
        negated = type(bpoly)(-bpoly.values, bpoly.weights)
    else:
        negated = type(bpoly)(-bpoly.c)
    result = minimize(negated, max_steps=max_steps, eps=eps)
    return OptimizeResult(f=-result.f, x=result.x)
