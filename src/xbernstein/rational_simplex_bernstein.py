r"""Positive-weight scalar rational Bernstein functions on simplices."""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from .rational_bernstein import RationalBernstein, _from_homogeneous
from .simplex_bernstein import (
    Bernstein2DS,
    Bernstein3DS,
    Bernstein4DS,
    _edge_subdivision_data,
    _elevate_coefficients,
    _evaluate_simplex_coefficients,
    _infer_degree,
    _multi_indices,
    _segment_coefficients,
)


def _evaluate_rational_simplex(
    homogeneous: Float[jax.Array, "..."],
    point: Float[jax.Array, "..."],
    dimensions: int,
) -> Float[jax.Array, "..."]:
    numerator = _evaluate_simplex_coefficients(
        homogeneous[0], point, dimensions
    )
    denominator = _evaluate_simplex_coefficients(
        homogeneous[1], point, dimensions
    )
    return numerator / denominator


def _rational_simplex_minimize(
    homogeneous: Float[jax.Array, "..."],
    dimensions: int,
    max_steps: int,
    eps: float,
) -> tuple[Float[jax.Array, ""], Float[jax.Array, "barycentric"]]:
    degree = _infer_degree(dimensions, homogeneous.shape[-1])
    barycentric_dimensions = dimensions + 1
    capacity = 1 + max_steps
    identity = jnp.eye(barycentric_dimensions, dtype=homogeneous.dtype)
    centroid = jnp.full(
        (1, barycentric_dimensions),
        1.0 / barycentric_dimensions,
        dtype=homogeneous.dtype,
    )
    edge_midpoints = jnp.asarray(
        [
            (identity[first] + identity[second]) / 2
            for first in range(barycentric_dimensions)
            for second in range(first + 1, barycentric_dimensions)
        ]
    )
    samples = jnp.concatenate((identity, edge_midpoints, centroid))
    sample_values = jax.vmap(
        lambda point: _evaluate_rational_simplex(
            homogeneous, point, dimensions
        )
    )(samples)
    best_index = jnp.argmin(sample_values)
    controls = (
        jnp.zeros(
            (capacity,) + homogeneous.shape, dtype=homogeneous.dtype
        )
        .at[0]
        .set(homogeneous)
    )
    vertices = (
        jnp.zeros(
            (capacity, barycentric_dimensions, barycentric_dimensions),
            dtype=homogeneous.dtype,
        )
        .at[0]
        .set(identity)
    )
    values = homogeneous[0] / homogeneous[1]
    bounds = (
        jnp.full(capacity, jnp.inf, dtype=homogeneous.dtype)
        .at[0]
        .set(jnp.min(values))
    )
    edges_np, vertex_maps_np, matrices_np = _edge_subdivision_data(
        dimensions, degree
    )
    edges = jnp.asarray(edges_np)
    vertex_maps = jnp.asarray(vertex_maps_np, dtype=homogeneous.dtype)
    matrices = jnp.asarray(matrices_np, dtype=homogeneous.dtype)
    state = (
        controls,
        vertices,
        bounds,
        sample_values[best_index],
        samples[best_index],
        jnp.asarray(0),
    )

    def condition(current):
        _, _, current_bounds, value, _, step = current
        return (step < max_steps) & ((value - jnp.min(current_bounds)) > eps)

    def body(current):
        controls, vertices, bounds, value, point, step = current
        selected = jnp.argmin(bounds)
        current_homogeneous = controls[selected]
        current_vertices = vertices[selected]
        local_values = jax.vmap(
            lambda candidate: _evaluate_rational_simplex(
                current_homogeneous, candidate, dimensions
            )
        )(samples)
        local_index = jnp.argmin(local_values)
        candidate_value = local_values[local_index]
        candidate_point = samples[local_index] @ current_vertices
        replace = candidate_value < value
        value = jnp.where(replace, candidate_value, value)
        point = jnp.where(replace, candidate_point, point)

        edge_vectors = (
            current_vertices[edges[:, 0]] - current_vertices[edges[:, 1]]
        )
        edge = jnp.argmax(jnp.sum(edge_vectors * edge_vectors, axis=-1))
        selected_vertex_maps = vertex_maps[edge]
        selected_matrices = matrices[edge]
        child_controls = jnp.einsum(
            "joi,ki->jko", selected_matrices, current_homogeneous
        )
        child_vertices = jnp.einsum(
            "jri,ik->jrk", selected_vertex_maps, current_vertices
        )
        child_values = child_controls[:, 0] / child_controls[:, 1]
        controls = controls.at[selected].set(child_controls[0])
        vertices = vertices.at[selected].set(child_vertices[0])
        bounds = bounds.at[selected].set(jnp.min(child_values[0]))
        new_index = step + 1
        controls = controls.at[new_index].set(child_controls[1])
        vertices = vertices.at[new_index].set(child_vertices[1])
        bounds = bounds.at[new_index].set(jnp.min(child_values[1]))
        return controls, vertices, bounds, value, point, step + 1

    result = jax.lax.while_loop(condition, body, state)
    return result[3], result[4]


def _make_rational_minimizer(dimensions: int):
    @jax.custom_jvp
    def minimize(
        homogeneous: Float[jax.Array, "..."],
        max_steps: int = 200,
        eps: float = 1e-6,
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, "barycentric"]]:
        return _rational_simplex_minimize(
            homogeneous, dimensions, max_steps, eps
        )

    @minimize.defjvp
    def minimize_jvp(
        primals: tuple[Float[jax.Array, "..."], int, float],
        tangents: tuple[Float[jax.Array, "..."], int, float],
    ) -> tuple[tuple[Float[jax.Array, ""], Float[jax.Array, "barycentric"]], tuple[Float[jax.Array, ""], Float[jax.Array, "barycentric"]]]:
        homogeneous, max_steps, eps = primals
        tangent_homogeneous, _, _ = tangents
        value, point = _rational_simplex_minimize(
            homogeneous, dimensions, max_steps, eps
        )
        tangent_value = jax.jvp(
            lambda h: _evaluate_rational_simplex(h, point, dimensions),
            (homogeneous,),
            (tangent_homogeneous,),
        )[1]

        def chart(y):
            return jnp.concatenate((y, jnp.atleast_1d(1.0 - jnp.sum(y))))

        y = point[:-1]

        def gradient(h):
            return jax.grad(
                lambda coordinates: _evaluate_rational_simplex(
                    h, chart(coordinates), dimensions
                )
            )(y)

        tangent_gradient = jax.jvp(
            gradient, (homogeneous,), (tangent_homogeneous,)
        )[1]
        hessian = jax.hessian(
            lambda coordinates: _evaluate_rational_simplex(
                homogeneous, chart(coordinates), dimensions
            )
        )(y)
        tangent_y = jax.lax.cond(
            jnp.any(point == 0.0),
            lambda _: jnp.zeros_like(y),
            lambda _: -jnp.linalg.solve(hessian, tangent_gradient),
            operand=None,
        )
        tangent_point = jnp.concatenate(
            (tangent_y, jnp.atleast_1d(-jnp.sum(tangent_y)))
        )
        return (value, point), (tangent_value, tangent_point)

    return minimize


_minimize_rational_2ds = _make_rational_minimizer(2)
_minimize_rational_3ds = _make_rational_minimizer(3)
_minimize_rational_4ds = _make_rational_minimizer(4)


class _RationalSimplexBernstein(eqx.Module):
    r"""Represent a positive-weight rational function on a standard simplex.

    For packed degree-$n$ controls and positive weights,

    $$
    R(\boldsymbol\lambda)=
    \frac{\sum_{|\alpha|=n}w_\alpha c_\alpha B_\alpha^n}
         {\sum_{|\alpha|=n}w_\alpha B_\alpha^n}.
    $$

    The homogeneous array has shape ``(*batch, 2, coefficient_count)``;
    component zero stores $w_\alpha c_\alpha$ and component one stores
    $w_\alpha$. Leading axes of ``values`` and ``weights`` broadcast, while
    their packed coefficient-axis lengths must agree.

    Batch dimensions
    ----------------

    Every axis before the homogeneous and packed coefficient axes is a batch
    axis. ``values`` and ``weights`` right-broadcast on those axes. Arithmetic
    broadcasts the batch shapes separately from degree alignment, while
    coordinate arrays create evaluation axes after the batch axes.

    Addition, subtraction, and multiplication use denominator products and
    therefore add total degrees. Tensor-only operations such as axis
    ``split``, ``slice``, ``int``, and ``integrate_out`` are not defined for
    simplex functions.
    """

    h: Float[jax.Array, "... component coefficient"]
    simplex_dimensions: ClassVar[int]
    polynomial_type: ClassVar[type]

    def __init__(
        self,
        values: Float[jax.Array, "*value_batch coefficient"],
        weights: Float[jax.Array, "*weight_batch coefficient"],
    ):
        values = jnp.asarray(values)
        weights = jnp.asarray(weights)
        if values.ndim < 1 or weights.ndim < 1:
            raise ValueError(
                "values and weights must have a packed coefficient axis"
            )
        if values.shape[-1] != weights.shape[-1]:
            raise ValueError(
                "values and weights must have the same packed coefficient count"
            )
        _infer_degree(self.simplex_dimensions, values.shape[-1])
        dtype = jnp.result_type(values, weights, jnp.float32)
        values = values.astype(dtype)
        weights = weights.astype(dtype)
        batch_shape = jnp.broadcast_shapes(
            values.shape[:-1], weights.shape[:-1]
        )
        values = jnp.broadcast_to(values, batch_shape + (values.shape[-1],))
        weights = jnp.broadcast_to(
            weights, batch_shape + (weights.shape[-1],)
        )
        weights = eqx.error_if(
            weights,
            jnp.any(weights <= 0.0),
            f"{type(self).__name__} weights must be strictly positive",
        )
        self.h = jnp.stack((weights * values, weights), axis=-2)

    @classmethod
    def _from_homogeneous(
        cls,
        numerator: Float[jax.Array, "*batch coefficient"],
        denominator: Float[jax.Array, "*batch coefficient"],
    ):
        result = object.__new__(cls)
        object.__setattr__(
            result, "h", jnp.stack((numerator, denominator), axis=-2)
        )
        return result

    @property
    def numerator(self):
        r"""Return $N=\sum_\alpha w_\alpha c_\alpha B_\alpha^n$."""
        return self.polynomial_type(self.h[..., 0, :])

    @property
    def denominator(self):
        r"""Return $D=\sum_\alpha w_\alpha B_\alpha^n>0$."""
        return self.polynomial_type(self.h[..., 1, :])

    @property
    def values(self) -> Float[jax.Array, "*batch coefficient"]:
        r"""Return dehomogenized controls $c_\alpha=h_{0,\alpha}/h_{1,\alpha}$."""
        return self.h[..., 0, :] / self.h[..., 1, :]

    @property
    def weights(self) -> Float[jax.Array, "*batch coefficient"]:
        r"""Return strictly positive controls $w_\alpha=h_{1,\alpha}$."""
        return self.h[..., 1, :]

    @property
    def c(self) -> Float[jax.Array, "*batch coefficient"]:
        """Return :attr:`values`."""
        return self.values

    @property
    def w(self) -> Float[jax.Array, "*batch coefficient"]:
        """Return :attr:`weights`."""
        return self.weights

    @property
    def shape(self) -> tuple[int, ...]:
        """Return leading batch axes."""
        return self.h.shape[:-2]

    @property
    def order(self) -> int:
        """Return the scalar total degree $n$."""
        return _infer_degree(self.simplex_dimensions, self.h.shape[-1])

    @property
    def multi_indices(self) -> Int[jax.Array, "coefficient barycentric"]:
        """Return packed barycentric multi-indices."""
        return jnp.asarray(_multi_indices(self.simplex_dimensions, self.order))

    @property
    def dtype(self) -> str:
        """Return the homogeneous scalar dtype."""
        return str(self.h.dtype)

    def __call__(
        self, *coordinates: Float[jax.Array, "..."]
    ) -> Float[jax.Array, "..."]:
        r"""Evaluate $R=N/D$ at $d+1$ barycentric coordinates.

        Coordinate arrays broadcast to a shared point shape appended after
        ``self.shape``.
        """
        return self.numerator(*coordinates) / self.denominator(*coordinates)

    def _check_other(self, other: object):
        if not isinstance(other, type(self)):
            raise TypeError(
                f"{type(self).__name__} arithmetic requires another "
                f"{type(self).__name__}"
            )
        return other

    def __add__(self, other: object):
        r"""Return $(N_1D_2+N_2D_1)/(D_1D_2)$ exactly."""
        other = self._check_other(other)
        denominator = self.denominator * other.denominator
        numerator = (
            self.numerator * other.denominator
            + other.numerator * self.denominator
        )
        return type(self)._from_homogeneous(numerator.c, denominator.c)

    def __sub__(self, other: object):
        r"""Return $(N_1D_2-N_2D_1)/(D_1D_2)$ exactly."""
        other = self._check_other(other)
        denominator = self.denominator * other.denominator
        numerator = (
            self.numerator * other.denominator
            - other.numerator * self.denominator
        )
        return type(self)._from_homogeneous(numerator.c, denominator.c)

    def __mul__(self, other: object):
        r"""Return $(N_1N_2)/(D_1D_2)$ exactly."""
        other = self._check_other(other)
        numerator = self.numerator * other.numerator
        denominator = self.denominator * other.denominator
        return type(self)._from_homogeneous(numerator.c, denominator.c)

    def deriv(self, m: int = 1, axis: int = 0):
        r"""Return the ambient derivative $\partial_{\lambda_a}^mR$.

        Each step applies $(N_aD-ND_a)/D^2$ and degree-elevates the numerator
        to the denominator's total degree.
        """
        self.numerator.deriv(m=0, axis=axis)
        if m < 0:
            raise ValueError("derivative order must be non-negative")
        result = self
        for _ in range(m):
            numerator = (
                result.numerator.deriv(axis=axis) * result.denominator
                - result.numerator * result.denominator.deriv(axis=axis)
            )
            denominator = result.denominator * result.denominator
            numerator_coefficients = _elevate_coefficients(
                numerator.c,
                self.simplex_dimensions,
                denominator.order,
            )
            result = type(self)._from_homogeneous(
                numerator_coefficients, denominator.c
            )
        return result

    def weight_sensitivity(self) -> "RationalSimplexBernstein":
        r"""Return sensitivities to every packed simplex control weight.

        The added final batch axis selects the differentiated packed
        coefficient.  With dehomogenized controls fixed, it represents

        $$
        \frac{\partial R}{\partial w_i}
        = \frac{B_i(c_iD-N)}{D^2}.
        $$

        The return type matches ``self`` and has twice the total degree.
        """
        count = self.h.shape[-1]
        numerator = self.numerator
        denominator = self.denominator
        basis = self.polynomial_type(jnp.eye(count, dtype=self.h.dtype))
        difference = self.polynomial_type(
            self.values[..., :, None] * denominator.c[..., None, :]
            - numerator.c[..., None, :]
        )
        sensitivity_numerator = basis * difference
        sensitivity_denominator = denominator * denominator
        denominator_coefficients = jnp.broadcast_to(
            sensitivity_denominator.c[..., None, :],
            sensitivity_numerator.c.shape,
        )
        return type(self)._from_homogeneous(
            sensitivity_numerator.c, denominator_coefficients
        )

    def segment(
        self,
        start: Float[jax.Array, "..."],
        end: Float[jax.Array, "..."],
    ) -> RationalBernstein:
        r"""Restrict to the barycentric segment from ``start`` to ``end``."""
        start = jnp.asarray(start, dtype=self.h.dtype)
        end = jnp.asarray(end, dtype=self.h.dtype)
        numerator = _segment_coefficients(
            self.h[..., 0, :], start, end, self.simplex_dimensions
        )
        denominator = _segment_coefficients(
            self.h[..., 1, :], start, end, self.simplex_dimensions
        )
        return _from_homogeneous(numerator, denominator)


class RationalBernstein2DS(_RationalSimplexBernstein):
    r"""Represent a positive-weight rational Bernstein function on a triangle.

    Evaluation accepts three barycentric coordinates. Degree-$n$ ``values``
    and ``weights`` have packed final length $\binom{n+2}{2}$.
    """

    simplex_dimensions: ClassVar[int] = 2
    polynomial_type: ClassVar[type] = Bernstein2DS


class RationalBernstein3DS(_RationalSimplexBernstein):
    r"""Represent a positive-weight rational Bernstein function on a tetrahedron.

    Evaluation accepts four barycentric coordinates. Degree-$n$ packed arrays
    have final length $\binom{n+3}{3}$.
    """

    simplex_dimensions: ClassVar[int] = 3
    polynomial_type: ClassVar[type] = Bernstein3DS


class RationalBernstein4DS(_RationalSimplexBernstein):
    r"""Represent a positive-weight rational Bernstein function on a 4-simplex.

    Evaluation accepts five barycentric coordinates. Degree-$n$ packed arrays
    have final length $\binom{n+4}{4}$.
    """

    simplex_dimensions: ClassVar[int] = 4
    polynomial_type: ClassVar[type] = Bernstein4DS
