r"""Positive-weight scalar rational tensor-product Bernstein functions."""

import itertools
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float

from ._tensor_bernstein import (
    _evaluate_tensor_coefficients,
    _split_tensor_coefficients,
)
from .bernstein_2d import Bernstein2D
from .bernstein_3d import Bernstein3D
from .bernstein_4d import Bernstein4D
from .rational_bernstein import RationalBernstein, _from_homogeneous


def _homogeneous_component(h: jax.Array, dimensions: int, index: int) -> jax.Array:
    return jnp.take(h, index, axis=-(dimensions + 1))


class _RationalTensorBernstein(eqx.Module):
    """Implement shared positive-weight rational tensor-product operations."""

    h: Float[jax.Array, "..."]
    parameter_dimensions: ClassVar[int]
    polynomial_type: ClassVar[type]

    def __init__(self, values, weights):
        dimensions = self.parameter_dimensions
        values = jnp.asarray(values)
        weights = jnp.asarray(weights)
        if values.ndim < dimensions or weights.ndim < dimensions:
            raise ValueError(
                f"values and weights must have {dimensions} coefficient axes"
            )
        if values.shape[-dimensions:] != weights.shape[-dimensions:]:
            raise ValueError(
                "values and weights must have the same coefficient-axis shape"
            )

        dtype = jnp.result_type(values, weights, jnp.float32)
        values = values.astype(dtype)
        weights = weights.astype(dtype)
        degree_shape = values.shape[-dimensions:]
        batch_shape = jnp.broadcast_shapes(
            values.shape[:-dimensions], weights.shape[:-dimensions]
        )
        values = jnp.broadcast_to(values, batch_shape + degree_shape)
        weights = jnp.broadcast_to(weights, batch_shape + degree_shape)
        weights = eqx.error_if(
            weights,
            jnp.any(weights <= 0.0),
            f"{type(self).__name__} weights must be strictly positive",
        )
        self.h = jnp.stack((weights * values, weights), axis=-(dimensions + 1))

    @classmethod
    def _from_homogeneous(cls, numerator: jax.Array, denominator: jax.Array):
        result = object.__new__(cls)
        axis = -(cls.parameter_dimensions + 1)
        object.__setattr__(result, "h", jnp.stack((numerator, denominator), axis=axis))
        return result

    @property
    def values(self) -> jax.Array:
        """Return the dehomogenized tensor control values."""
        return self.numerator.c / self.denominator.c

    @property
    def weights(self) -> jax.Array:
        """Return the positive tensor weights."""
        return self.denominator.c

    @property
    def c(self) -> jax.Array:
        """Return :attr:`values` using the conventional short name."""
        return self.values

    @property
    def w(self) -> jax.Array:
        """Return :attr:`weights` using the conventional short name."""
        return self.weights

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the leading batch shape."""
        return self.h.shape[: -(self.parameter_dimensions + 1)]

    @property
    def order(self) -> jax.Array:
        """Return the parameter-wise degree vector."""
        return jnp.asarray(self.h.shape[-self.parameter_dimensions :]) - 1

    @property
    def dtype(self) -> str:
        """Return the homogeneous coefficient dtype."""
        return str(self.h.dtype)

    @property
    def numerator(self):
        """Return the tensor Bernstein numerator."""
        return self.polynomial_type(
            _homogeneous_component(self.h, self.parameter_dimensions, 0)
        )

    @property
    def denominator(self):
        """Return the positive tensor Bernstein denominator."""
        return self.polynomial_type(
            _homogeneous_component(self.h, self.parameter_dimensions, 1)
        )

    def __call__(self, *ts):
        """Evaluate the rational tensor function at ``ts``."""
        return self.numerator(*ts) / self.denominator(*ts)

    def _check_other(self, other: object):
        if not isinstance(other, type(self)):
            raise TypeError(
                f"{type(self).__name__} arithmetic requires another "
                f"{type(self).__name__}"
            )
        return other

    def __add__(self, other: object):
        """Return the exact pointwise sum."""
        other = self._check_other(other)
        denominator = self.denominator * other.denominator
        numerator = (
            self.numerator * other.denominator
            + other.numerator * self.denominator
        )
        return type(self)._from_homogeneous(numerator.c, denominator.c)

    def __sub__(self, other: object):
        """Return the exact pointwise difference."""
        other = self._check_other(other)
        denominator = self.denominator * other.denominator
        numerator = (
            self.numerator * other.denominator
            - other.numerator * self.denominator
        )
        return type(self)._from_homogeneous(numerator.c, denominator.c)

    def __mul__(self, other: object):
        """Return the exact pointwise product."""
        other = self._check_other(other)
        numerator = self.numerator * other.numerator
        denominator = self.denominator * other.denominator
        return type(self)._from_homogeneous(numerator.c, denominator.c)

    def deriv(self, m: int = 1, axis: int = 0):
        """Return the exact ``m``-th partial derivative along ``axis``."""
        self.numerator._axis_index(axis)
        if m < 0:
            raise ValueError("derivative order must be non-negative")
        result = self
        for _ in range(m):
            numerator = (
                result.numerator.deriv(axis=axis) * result.denominator
                - result.numerator * result.denominator.deriv(axis=axis)
            )
            denominator = result.denominator * result.denominator
            target_degrees = tuple(int(value) for value in denominator.order)
            numerator_coefficients = numerator._elevate(
                numerator.c, target_degrees
            )
            result = type(self)._from_homogeneous(
                numerator_coefficients, denominator.c
            )
        return result

    def split(self, t: jax.Array | float = jnp.array(0.5), axis: int = 0):
        """Split and reparameterize one tensor parameter axis."""
        numerator_left, numerator_right = self.numerator.split(t=t, axis=axis)
        denominator_left, denominator_right = self.denominator.split(t=t, axis=axis)
        return (
            type(self)._from_homogeneous(
                numerator_left.c, denominator_left.c
            ),
            type(self)._from_homogeneous(
                numerator_right.c, denominator_right.c
            ),
        )

    def _lower_dimension(self, numerator: jax.Array, denominator: jax.Array):
        if self.parameter_dimensions == 2:
            return _from_homogeneous(numerator, denominator)
        if self.parameter_dimensions == 3:
            return RationalBernstein2D._from_homogeneous(numerator, denominator)
        if self.parameter_dimensions == 4:
            return RationalBernstein3D._from_homogeneous(numerator, denominator)
        raise RuntimeError("rational tensor functions require at least 2 dimensions")

    def slice(self, value: jax.Array | float, axis: int = 0):
        """Fix one parameter and return the lower-dimensional rational function."""
        numerator = self.numerator.slice(value=value, axis=axis)
        denominator = self.denominator.slice(value=value, axis=axis)
        return self._lower_dimension(numerator.c, denominator.c)

    def segment(self, start: jax.Array, end: jax.Array) -> RationalBernstein:
        """Restrict the rational tensor function to an affine segment."""
        numerator = self.numerator.segment(start, end)
        denominator = self.denominator.segment(start, end)
        return _from_homogeneous(numerator.c, denominator.c)


class RationalBernstein2D(_RationalTensorBernstein):
    """Represent a positive-weight scalar rational Bernstein function on $[0,1]^2$."""

    parameter_dimensions: ClassVar[int] = 2
    polynomial_type: ClassVar[type] = Bernstein2D

    def __init__(self, values, weights):
        super().__init__(values, weights)


class RationalBernstein3D(_RationalTensorBernstein):
    """Represent a positive-weight scalar rational Bernstein function on $[0,1]^3$."""

    parameter_dimensions: ClassVar[int] = 3
    polynomial_type: ClassVar[type] = Bernstein3D

    def __init__(self, values, weights):
        super().__init__(values, weights)


class RationalBernstein4D(_RationalTensorBernstein):
    """Represent a positive-weight scalar rational Bernstein function on $[0,1]^4$."""

    parameter_dimensions: ClassVar[int] = 4
    polynomial_type: ClassVar[type] = Bernstein4D

    def __init__(self, values, weights):
        super().__init__(values, weights)


def _evaluate_rational_tensor(h: jax.Array, point: jax.Array) -> jax.Array:
    numerator = _evaluate_tensor_coefficients(h[0], point)
    denominator = _evaluate_tensor_coefficients(h[1], point)
    return numerator / denominator


def _split_rational_tensor(
    h: jax.Array, axis: jax.Array
) -> tuple[jax.Array, jax.Array]:
    numerator_left, numerator_right = _split_tensor_coefficients(h[0], axis)
    denominator_left, denominator_right = _split_tensor_coefficients(h[1], axis)
    return (
        jnp.stack((numerator_left, denominator_left)),
        jnp.stack((numerator_right, denominator_right)),
    )


@jax.custom_jvp
def _minimize(h: jax.Array, max_steps: int = 200, eps: float = 1e-6):
    """Globally minimize one unbatched positive-weight rational tensor."""
    dimensions = h.ndim - 1
    capacity = max_steps + 1
    corners = jnp.asarray(list(itertools.product((0, 1), repeat=dimensions)))
    samples = jnp.concatenate(
        [corners, jnp.full((1, dimensions), 0.5, dtype=h.dtype)]
    )
    initial_values = jax.vmap(lambda point: _evaluate_rational_tensor(h, point))(
        samples
    )
    initial_index = jnp.argmin(initial_values)
    lower = jnp.zeros((capacity, dimensions), dtype=h.dtype)
    upper = jnp.ones((capacity, dimensions), dtype=h.dtype)
    control = jnp.zeros((capacity,) + h.shape, dtype=h.dtype).at[0].set(h)
    values = h[0] / h[1]
    bounds = jnp.full(capacity, jnp.inf, dtype=h.dtype).at[0].set(jnp.min(values))
    state = (
        lower,
        upper,
        control,
        bounds,
        initial_values[initial_index],
        samples[initial_index],
        0,
    )

    def condition(current):
        _, _, _, current_bounds, value, _, step = current
        return (step < max_steps) & ((value - jnp.min(current_bounds)) > eps)

    def body(current):
        lower, upper, control, bounds, value, point, step = current
        index = jnp.argmin(bounds)
        lo, hi, current_h = lower[index], upper[index], control[index]
        axis = jnp.argmax(hi - lo)
        left, right = _split_rational_tensor(current_h, axis)
        midpoint = 0.5 * (lo + hi)
        candidate_points = lo + samples * (hi - lo)
        candidate_values = jax.vmap(
            lambda candidate: _evaluate_rational_tensor(current_h, candidate)
        )(samples)
        candidate_index = jnp.argmin(candidate_values)
        candidate_value = candidate_values[candidate_index]
        replace = candidate_value < value
        value = jnp.where(replace, candidate_value, value)
        point = jnp.where(replace, candidate_points[candidate_index], point)
        next_index = step + 1
        lower = (
            lower.at[index]
            .set(lo)
            .at[next_index]
            .set(lo.at[axis].set(midpoint[axis]))
        )
        upper = (
            upper.at[index]
            .set(hi.at[axis].set(midpoint[axis]))
            .at[next_index]
            .set(hi)
        )
        control = control.at[index].set(left).at[next_index].set(right)
        left_values = left[0] / left[1]
        right_values = right[0] / right[1]
        bounds = (
            bounds.at[index]
            .set(jnp.min(left_values))
            .at[next_index]
            .set(jnp.min(right_values))
        )
        return lower, upper, control, bounds, value, point, step + 1

    result = jax.lax.while_loop(condition, body, state)
    return result[4], result[5]


@_minimize.defjvp
def _minimize_jvp(primals, tangents):
    h, max_steps, eps = primals
    tangent_h, _, _ = tangents
    primal = _minimize(h, max_steps=max_steps, eps=eps)
    point = primal[1]
    tangent_value = jax.jvp(
        lambda homogeneous: _evaluate_rational_tensor(homogeneous, point),
        (h,),
        (tangent_h,),
    )[1]

    def gradient(homogeneous):
        return jax.grad(
            lambda parameters: _evaluate_rational_tensor(
                homogeneous, parameters
            )
        )(point)

    tangent_gradient = jax.jvp(gradient, (h,), (tangent_h,))[1]
    hessian = jax.hessian(
        lambda parameters: _evaluate_rational_tensor(h, parameters)
    )(point)
    tangent_point = jax.lax.cond(
        jnp.any((point == 0.0) | (point == 1.0)),
        lambda _: jnp.zeros_like(point),
        lambda _: -jnp.linalg.solve(hessian, tangent_gradient),
        operand=None,
    )
    return primal, (tangent_value, tangent_point)
