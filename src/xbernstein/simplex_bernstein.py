r"""Scalar Bernstein polynomials on standard simplices."""

import functools
import itertools
import math
from typing import ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from .bernstein import Bernstein


@functools.lru_cache
def _multi_indices(dimensions: int, degree: int) -> tuple[tuple[int, ...], ...]:
    """Return packed barycentric indices in descending lexicographic order."""
    return tuple(
        sorted(
            (
                alpha
                for alpha in itertools.product(
                    range(degree + 1), repeat=dimensions + 1
                )
                if sum(alpha) == degree
            ),
            reverse=True,
        )
    )


def _infer_degree(dimensions: int, coefficient_count: int) -> int:
    if coefficient_count < 1:
        raise ValueError("the packed coefficient axis must be non-empty")
    degree = 0
    while math.comb(degree + dimensions, dimensions) < coefficient_count:
        degree += 1
    if math.comb(degree + dimensions, dimensions) != coefficient_count:
        raise ValueError(
            f"packed coefficient count {coefficient_count} is invalid for a "
            f"{dimensions}D simplex"
        )
    return degree


def _multinomial(degree: int, alpha: tuple[int, ...]) -> int:
    result = math.factorial(degree)
    for value in alpha:
        result //= math.factorial(value)
    return result


@functools.lru_cache
def _elevation_matrix(
    dimensions: int, degree: int, target_degree: int
) -> np.ndarray:
    source = _multi_indices(dimensions, degree)
    target = _multi_indices(dimensions, target_degree)
    extra_degree = target_degree - degree
    matrix = np.zeros((len(target), len(source)))
    for row, beta in enumerate(target):
        denominator = _multinomial(target_degree, beta)
        for column, alpha in enumerate(source):
            delta = tuple(b - a for a, b in zip(alpha, beta))
            if min(delta) < 0 or sum(delta) != extra_degree:
                continue
            matrix[row, column] = (
                _multinomial(degree, alpha)
                * _multinomial(extra_degree, delta)
                / denominator
            )
    return matrix


def _elevate_coefficients(
    coefficients: jax.Array, dimensions: int, target_degree: int
) -> jax.Array:
    degree = _infer_degree(dimensions, coefficients.shape[-1])
    if target_degree < degree:
        raise ValueError("target degree must not be smaller than the current degree")
    if target_degree == degree:
        return coefficients
    matrix = jnp.asarray(
        _elevation_matrix(dimensions, degree, target_degree),
        dtype=coefficients.dtype,
    )
    return jnp.einsum("oi,...i->...o", matrix, coefficients)


def _evaluate_simplex_coefficients(
    coefficients: jax.Array, point: jax.Array, dimensions: int
) -> jax.Array:
    """Evaluate one unbatched packed polynomial at one barycentric point."""
    degree = _infer_degree(dimensions, coefficients.shape[-1])
    indices = jnp.asarray(_multi_indices(dimensions, degree))
    scales = jnp.asarray(
        [_multinomial(degree, alpha) for alpha in _multi_indices(dimensions, degree)],
        dtype=coefficients.dtype,
    )
    basis = scales * jnp.prod(point[None, :] ** indices, axis=-1)
    return jnp.sum(coefficients * basis)


@functools.lru_cache
def _product_data(dimensions: int, left_degree: int, right_degree: int):
    left = _multi_indices(dimensions, left_degree)
    right = _multi_indices(dimensions, right_degree)
    output = _multi_indices(dimensions, left_degree + right_degree)
    output_lookup = {alpha: index for index, alpha in enumerate(output)}
    indices = []
    scales = []
    for alpha in left:
        for beta in right:
            gamma = tuple(a + b for a, b in zip(alpha, beta))
            indices.append(output_lookup[gamma])
            scales.append(
                _multinomial(left_degree, alpha)
                * _multinomial(right_degree, beta)
                / _multinomial(left_degree + right_degree, gamma)
            )
    return np.asarray(indices), np.asarray(scales)


def _multiply_coefficients(
    left: jax.Array, right: jax.Array, dimensions: int
) -> jax.Array:
    left_degree = _infer_degree(dimensions, left.shape[-1])
    right_degree = _infer_degree(dimensions, right.shape[-1])
    batch_shape = jnp.broadcast_shapes(left.shape[:-1], right.shape[:-1])
    left = jnp.broadcast_to(left, batch_shape + (left.shape[-1],))
    right = jnp.broadcast_to(right, batch_shape + (right.shape[-1],))
    output_count = len(_multi_indices(dimensions, left_degree + right_degree))
    flat_indices, scales = _product_data(
        dimensions, left_degree, right_degree
    )
    products = (
        left[..., :, None]
        * right[..., None, :]
        * jnp.asarray(scales, dtype=left.dtype).reshape(
            (1,) * len(batch_shape) + (left.shape[-1], right.shape[-1])
        )
    )
    result = jnp.zeros(batch_shape + (output_count,), dtype=products.dtype)
    return result.at[..., jnp.asarray(flat_indices).reshape(
        left.shape[-1], right.shape[-1]
    )].add(products)


@functools.lru_cache
def _segment_data(dimensions: int, degree: int, output_index: int):
    source = _multi_indices(dimensions, degree)
    lookup = {alpha: index for index, alpha in enumerate(source)}
    beta_indices = _multi_indices(dimensions, output_index)
    gamma_indices = _multi_indices(dimensions, degree - output_index)
    source_indices = []
    betas = []
    gammas = []
    scales = []
    for beta in beta_indices:
        for gamma in gamma_indices:
            alpha = tuple(b + g for b, g in zip(beta, gamma))
            source_indices.append(lookup[alpha])
            betas.append(beta)
            gammas.append(gamma)
            scales.append(
                _multinomial(output_index, beta)
                * _multinomial(degree - output_index, gamma)
            )
    return (
        np.asarray(source_indices),
        np.asarray(betas),
        np.asarray(gammas),
        np.asarray(scales),
    )


def _segment_coefficients(
    coefficients: jax.Array,
    start: jax.Array,
    end: jax.Array,
    dimensions: int,
) -> jax.Array:
    barycentric_dimensions = dimensions + 1
    if start.ndim < 1 or end.ndim < 1:
        raise ValueError("segment endpoints must have a final barycentric axis")
    if (
        start.shape[-1] != barycentric_dimensions
        or end.shape[-1] != barycentric_dimensions
    ):
        raise ValueError(
            f"segment endpoints must have final length {barycentric_dimensions}"
        )
    degree = _infer_degree(dimensions, coefficients.shape[-1])
    batch_shape = jnp.broadcast_shapes(
        coefficients.shape[:-1], start.shape[:-1], end.shape[:-1]
    )
    coefficients = jnp.broadcast_to(
        coefficients, batch_shape + (coefficients.shape[-1],)
    )
    start = jnp.broadcast_to(start, batch_shape + (barycentric_dimensions,))
    end = jnp.broadcast_to(end, batch_shape + (barycentric_dimensions,))
    output = []
    for index in range(degree + 1):
        source_indices, betas, gammas, scales = _segment_data(
            dimensions, degree, index
        )
        beta = jnp.asarray(betas)
        gamma = jnp.asarray(gammas)
        factors = (
            jnp.prod(end[..., None, :] ** beta, axis=-1)
            * jnp.prod(start[..., None, :] ** gamma, axis=-1)
            * jnp.asarray(scales, dtype=coefficients.dtype)
        )
        output.append(
            jnp.sum(coefficients[..., jnp.asarray(source_indices)] * factors, axis=-1)
        )
    return jnp.stack(output, axis=-1)


def _restriction_matrix(
    dimensions: int, degree: int, vertices: np.ndarray
) -> np.ndarray:
    indices = _multi_indices(dimensions, degree)
    lookup = {alpha: index for index, alpha in enumerate(indices)}
    matrix = np.zeros((len(indices), len(indices)))
    zero: tuple[int, ...] = (0,) * (dimensions + 1)
    for row, beta in enumerate(indices):
        distribution: dict[tuple[int, ...], float] = {zero: 1.0}
        for child_vertex, repetitions in enumerate(beta):
            for _ in range(repetitions):
                next_distribution: dict[tuple[int, ...], float] = {}
                for alpha, weight in distribution.items():
                    for parent_vertex in range(dimensions + 1):
                        next_alpha = list(alpha)
                        next_alpha[parent_vertex] += 1
                        next_alpha = tuple(next_alpha)
                        next_distribution[next_alpha] = (
                            next_distribution.get(next_alpha, 0.0)
                            + weight * vertices[child_vertex, parent_vertex]
                        )
                distribution = next_distribution
        for alpha, weight in distribution.items():
            matrix[row, lookup[alpha]] = weight
    return matrix


@functools.lru_cache
def _edge_subdivision_data(dimensions: int, degree: int):
    barycentric_dimensions = dimensions + 1
    edges = tuple(itertools.combinations(range(barycentric_dimensions), 2))
    vertex_maps = []
    matrices = []
    for first, second in edges:
        midpoint = np.zeros(barycentric_dimensions)
        midpoint[first] = 0.5
        midpoint[second] = 0.5
        left = np.eye(barycentric_dimensions)
        right = np.eye(barycentric_dimensions)
        left[second] = midpoint
        right[first] = midpoint
        maps = np.stack((left, right))
        vertex_maps.append(maps)
        matrices.append(
            np.stack(
                [
                    _restriction_matrix(dimensions, degree, child)
                    for child in maps
                ]
            )
        )
    return np.asarray(edges), np.asarray(vertex_maps), np.asarray(matrices)


def _simplex_minimize(
    coefficients: jax.Array, dimensions: int, max_steps: int, eps: float
) -> tuple[jax.Array, jax.Array]:
    degree = _infer_degree(dimensions, coefficients.shape[-1])
    barycentric_dimensions = dimensions + 1
    capacity = 1 + max_steps
    identity = jnp.eye(barycentric_dimensions, dtype=coefficients.dtype)
    centroid = jnp.full(
        (1, barycentric_dimensions),
        1.0 / barycentric_dimensions,
        dtype=coefficients.dtype,
    )
    edge_midpoints = jnp.asarray(
        [
            (identity[first] + identity[second]) / 2
            for first, second in itertools.combinations(
                range(barycentric_dimensions), 2
            )
        ]
    )
    samples = jnp.concatenate((identity, edge_midpoints, centroid))
    sample_values = jax.vmap(
        lambda point: _evaluate_simplex_coefficients(
            coefficients, point, dimensions
        )
    )(samples)
    best_index = jnp.argmin(sample_values)
    controls = (
        jnp.zeros((capacity, coefficients.shape[-1]), dtype=coefficients.dtype)
        .at[0]
        .set(coefficients)
    )
    vertices = (
        jnp.zeros(
            (capacity, barycentric_dimensions, barycentric_dimensions),
            dtype=coefficients.dtype,
        )
        .at[0]
        .set(identity)
    )
    bounds = (
        jnp.full(capacity, jnp.inf, dtype=coefficients.dtype)
        .at[0]
        .set(jnp.min(coefficients))
    )
    edges_np, vertex_maps_np, matrices_np = _edge_subdivision_data(
        dimensions, degree
    )
    edges = jnp.asarray(edges_np)
    vertex_maps = jnp.asarray(vertex_maps_np, dtype=coefficients.dtype)
    matrices = jnp.asarray(matrices_np, dtype=coefficients.dtype)
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
        current_coefficients = controls[selected]
        current_vertices = vertices[selected]
        local_values = jax.vmap(
            lambda candidate: _evaluate_simplex_coefficients(
                current_coefficients, candidate, dimensions
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
            "joi,i->jo", selected_matrices, current_coefficients
        )
        child_vertices = jnp.einsum(
            "jri,ik->jrk", selected_vertex_maps, current_vertices
        )
        controls = controls.at[selected].set(child_controls[0])
        vertices = vertices.at[selected].set(child_vertices[0])
        bounds = bounds.at[selected].set(jnp.min(child_controls[0]))
        new_index = step + 1
        controls = controls.at[new_index].set(child_controls[1])
        vertices = vertices.at[new_index].set(child_vertices[1])
        bounds = bounds.at[new_index].set(jnp.min(child_controls[1]))
        return controls, vertices, bounds, value, point, step + 1

    result = jax.lax.while_loop(condition, body, state)
    return result[3], result[4]


def _make_minimizer(dimensions: int):
    @jax.custom_jvp
    def minimize(
        coefficients: jax.Array, max_steps: int = 200, eps: float = 1e-6
    ):
        return _simplex_minimize(coefficients, dimensions, max_steps, eps)

    @minimize.defjvp
    def minimize_jvp(primals, tangents):
        coefficients, max_steps, eps = primals
        tangent_coefficients, _, _ = tangents
        value, point = _simplex_minimize(
            coefficients, dimensions, max_steps, eps
        )
        tangent_value = _evaluate_simplex_coefficients(
            tangent_coefficients, point, dimensions
        )

        def chart(y):
            return jnp.concatenate((y, jnp.atleast_1d(1.0 - jnp.sum(y))))

        y = point[:-1]
        tangent_gradient = jax.grad(
            lambda coordinates: _evaluate_simplex_coefficients(
                tangent_coefficients, chart(coordinates), dimensions
            )
        )(y)
        hessian = jax.hessian(
            lambda coordinates: _evaluate_simplex_coefficients(
                coefficients, chart(coordinates), dimensions
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


_minimize_2ds = _make_minimizer(2)
_minimize_3ds = _make_minimizer(3)
_minimize_4ds = _make_minimizer(4)


class _SimplexBernstein(eqx.Module):
    r"""Represent a scalar Bernstein polynomial on a standard simplex.

    A dimension-$d$ simplex uses $d+1$ barycentric coordinates
    $\boldsymbol\lambda$. Degree-$n$ coefficients are packed on the final
    array axis in descending lexicographic order of
    $\alpha\in\mathbb N^{d+1}$ with $|\alpha|=n$:

    $$
    p(\boldsymbol\lambda)=
    \sum_{|\alpha|=n}c_\alpha
    \binom{n}{\alpha}\boldsymbol\lambda^\alpha.
    $$

    All preceding axes are independent batch axes. Arithmetic broadcasts only
    those axes; evaluation coordinates broadcast into point axes appended
    after them.
    """

    c: Float[jax.Array, "... coefficient"]
    simplex_dimensions: ClassVar[int]

    def __init__(self, c):
        coefficients = jnp.asarray(c)
        if coefficients.ndim < 1:
            raise ValueError("coefficients must have a packed coefficient axis")
        _infer_degree(self.simplex_dimensions, coefficients.shape[-1])
        self.c = coefficients

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the leading batch shape, excluding the packed coefficient axis."""
        return self.c.shape[:-1]

    @property
    def order(self) -> int:
        """Return the scalar total degree $n$."""
        return _infer_degree(self.simplex_dimensions, self.c.shape[-1])

    @property
    def multi_indices(self) -> jax.Array:
        """Return packed barycentric multi-indices with shape ``(count, d+1)``."""
        return jnp.asarray(_multi_indices(self.simplex_dimensions, self.order))

    @property
    def dtype(self) -> str:
        """Return the coefficient scalar dtype."""
        return str(self.c.dtype)

    def _check_other(self, other: object):
        if not isinstance(other, type(self)):
            raise TypeError(
                f"{type(self).__name__} arithmetic requires another "
                f"{type(self).__name__}"
            )
        return other

    def _elevate(self, target_degree: int) -> jax.Array:
        return _elevate_coefficients(
            self.c, self.simplex_dimensions, target_degree
        )

    def __add__(self, other: Self) -> Self:
        r"""Return $p+q$ after elevating both operands to $\max(n,m)$."""
        other = self._check_other(other)
        target = max(self.order, other.order)
        batch_shape = jnp.broadcast_shapes(self.shape, other.shape)
        count = len(_multi_indices(self.simplex_dimensions, target))
        left = jnp.broadcast_to(self._elevate(target), batch_shape + (count,))
        right = jnp.broadcast_to(other._elevate(target), batch_shape + (count,))
        return type(self)(left + right)

    def __sub__(self, other: Self) -> Self:
        r"""Return $p-q$ after elevating both operands to $\max(n,m)$."""
        other = self._check_other(other)
        target = max(self.order, other.order)
        batch_shape = jnp.broadcast_shapes(self.shape, other.shape)
        count = len(_multi_indices(self.simplex_dimensions, target))
        left = jnp.broadcast_to(self._elevate(target), batch_shape + (count,))
        right = jnp.broadcast_to(other._elevate(target), batch_shape + (count,))
        return type(self)(left - right)

    def __mul__(self, other: Self) -> Self:
        r"""Return the degree-$(n+m)$ simplex Bernstein product $pq$."""
        other = self._check_other(other)
        return type(self)(
            _multiply_coefficients(
                self.c, other.c, self.simplex_dimensions
            )
        )

    def deriv(self, m: int = 1, axis: int = 0) -> Self:
        r"""Return the ambient barycentric derivative $\partial_{\lambda_a}^m p$.

        One derivative maps $c_\alpha$ at degree $n$ to
        $d_\beta=n\,c_{\beta+e_a}$ at degree $n-1$. Batch axes are preserved.
        """
        barycentric_dimensions = self.simplex_dimensions + 1
        if not 0 <= axis < barycentric_dimensions:
            raise ValueError(
                f"axis must be in [0, {barycentric_dimensions}), got {axis}"
            )
        if m < 0:
            raise ValueError("derivative order must be non-negative")
        coefficients = self.c
        degree = self.order
        if m > degree:
            return type(self)(
                jnp.zeros(coefficients.shape[:-1] + (1,), dtype=coefficients.dtype)
            )
        for current_degree in range(degree, degree - m, -1):
            source = _multi_indices(self.simplex_dimensions, current_degree)
            lookup = {alpha: index for index, alpha in enumerate(source)}
            target = _multi_indices(self.simplex_dimensions, current_degree - 1)
            source_indices = []
            for beta in target:
                alpha = list(beta)
                alpha[axis] += 1
                source_indices.append(lookup[tuple(alpha)])
            coefficients = current_degree * coefficients[
                ..., jnp.asarray(source_indices)
            ]
        return type(self)(coefficients)

    def __call__(self, *coordinates) -> jax.Array:
        r"""Evaluate at $d+1$ broadcast-compatible barycentric coordinates.

        On the standard simplex the coordinates are nonnegative and sum to
        one. Their broadcast point shape is appended after ``self.shape``.
        """
        expected = self.simplex_dimensions + 1
        if len(coordinates) != expected:
            raise TypeError(
                f"expected {expected} barycentric coordinate arrays, "
                f"got {len(coordinates)}"
            )
        parameters = tuple(
            jnp.asarray(value, dtype=self.c.dtype) for value in coordinates
        )
        point_shape = jnp.broadcast_shapes(*(value.shape for value in parameters))
        parameters = tuple(
            jnp.broadcast_to(value, point_shape) for value in parameters
        )
        point = jnp.stack(parameters, axis=-1)
        degree = self.order
        indices = jnp.asarray(self.multi_indices)
        scales = jnp.asarray(
            [
                _multinomial(degree, alpha)
                for alpha in _multi_indices(self.simplex_dimensions, degree)
            ],
            dtype=self.c.dtype,
        )
        basis = scales * jnp.prod(point[..., None, :] ** indices, axis=-1)
        coefficients = self.c.reshape(
            self.shape + (1,) * len(point_shape) + (self.c.shape[-1],)
        )
        basis = basis.reshape(
            (1,) * len(self.shape) + point_shape + (self.c.shape[-1],)
        )
        return jnp.sum(coefficients * basis, axis=-1)

    def segment(self, start: jax.Array, end: jax.Array) -> Bernstein:
        r"""Restrict to $(1-t)\,\mathrm{start}+t\,\mathrm{end}$.

        Endpoints use a final barycentric axis of length $d+1$ and their
        leading axes broadcast with ``self.shape``. The exact result has the
        same total degree as this simplex polynomial.
        """
        start = jnp.asarray(start, dtype=self.c.dtype)
        end = jnp.asarray(end, dtype=self.c.dtype)
        return Bernstein(
            _segment_coefficients(
                self.c, start, end, self.simplex_dimensions
            )
        )


class Bernstein2DS(_SimplexBernstein):
    r"""Represent a scalar Bernstein polynomial on a triangle.

    Evaluation takes three barycentric coordinates. A degree-$n$ packed
    coefficient axis has length $\binom{n+2}{2}$.
    """

    simplex_dimensions: ClassVar[int] = 2


class Bernstein3DS(_SimplexBernstein):
    r"""Represent a scalar Bernstein polynomial on a tetrahedron.

    Evaluation takes four barycentric coordinates. A degree-$n$ packed
    coefficient axis has length $\binom{n+3}{3}$.
    """

    simplex_dimensions: ClassVar[int] = 3


class Bernstein4DS(_SimplexBernstein):
    r"""Represent a scalar Bernstein polynomial on a 4-simplex.

    Evaluation takes five barycentric coordinates. A degree-$n$ packed
    coefficient axis has length $\binom{n+4}{4}$.
    """

    simplex_dimensions: ClassVar[int] = 4
