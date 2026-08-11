r"""One-dimensional scalar rational Bernstein functions on $[0,1]$."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float

from .bernstein import Bernstein, _elevate


def _from_homogeneous(
    numerator: jax.Array, denominator: jax.Array
) -> "RationalBernstein":
    result = object.__new__(RationalBernstein)
    object.__setattr__(result, "h", jnp.stack((numerator, denominator), axis=-2))
    return result


def _split_homogeneous(h: jax.Array, t: jax.Array | float):
    t = jnp.asarray(t, dtype=h.dtype)
    t = jnp.broadcast_to(t, h.shape[:-2])
    weight = t[..., None, None]
    work = h
    left = [work[..., 0]]
    right = [work[..., -1]]
    for _ in range(h.shape[-1] - 1):
        work = (1.0 - weight) * work[..., :-1] + weight * work[..., 1:]
        left.append(work[..., 0])
        right.append(work[..., -1])
    return jnp.stack(left, axis=-1), jnp.stack(right[::-1], axis=-1)


def _evaluate_homogeneous(h: jax.Array, t: jax.Array):
    numerator = Bernstein(h[..., 0, :])(t)
    denominator = Bernstein(h[..., 1, :])(t)
    return numerator / denominator


class RationalBernstein(eqx.Module):
    r"""Represent a scalar rational Bernstein function with positive weights.

    ``RationalBernstein(values, weights)`` represents

    $$
    r(t)=\frac{\sum_i w_i c_i B_i^n(t)}
               {\sum_i w_i B_i^n(t)}.
    $$

    The final axes of ``values`` and ``weights`` must have the same length.
    Their leading axes broadcast, and every weight must be strictly positive.
    Homogeneous coefficients are stored as ``h[..., 0, :] = weights * values``
    and ``h[..., 1, :] = weights``.
    """

    h: Float[jax.Array, "*batch 2 order"]

    def __init__(self, values, weights):
        values = jnp.asarray(values)
        weights = jnp.asarray(weights)
        if values.ndim == 0 or weights.ndim == 0:
            raise ValueError("values and weights must have a coefficient axis")
        if values.shape[-1] != weights.shape[-1]:
            raise ValueError(
                "values and weights must have the same coefficient-axis length"
            )

        dtype = jnp.result_type(values, weights, jnp.float32)
        values = values.astype(dtype)
        weights = weights.astype(dtype)
        batch_shape = jnp.broadcast_shapes(values.shape[:-1], weights.shape[:-1])
        coefficient_shape = (values.shape[-1],)
        values = jnp.broadcast_to(values, batch_shape + coefficient_shape)
        weights = jnp.broadcast_to(weights, batch_shape + coefficient_shape)
        weights = eqx.error_if(
            weights,
            jnp.any(weights <= 0.0),
            "RationalBernstein weights must be strictly positive",
        )
        self.h = jnp.stack((weights * values, weights), axis=-2)

    @property
    def values(self) -> jax.Array:
        """Return the dehomogenized control values."""
        return self.h[..., 0, :] / self.h[..., 1, :]

    @property
    def weights(self) -> jax.Array:
        """Return the positive rational weights."""
        return self.h[..., 1, :]

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
        """Return the homogeneous coefficient shape excluding its final axes."""
        return self.h.shape[:-2]

    @property
    def order(self) -> int:
        """Return the rational Bernstein degree."""
        return self.h.shape[-1] - 1

    @property
    def dtype(self) -> str:
        """Return the homogeneous coefficient dtype."""
        return str(self.h.dtype)

    @property
    def numerator(self) -> Bernstein:
        """Return the polynomial numerator."""
        return Bernstein(self.h[..., 0, :])

    @property
    def denominator(self) -> Bernstein:
        """Return the positive polynomial denominator."""
        return Bernstein(self.h[..., 1, :])

    def __call__(self, t):
        """Evaluate the rational function at ``t``."""
        return _evaluate_homogeneous(self.h, t)

    def _check_other(self, other: object) -> "RationalBernstein":
        if not isinstance(other, type(self)):
            raise TypeError(
                f"{type(self).__name__} arithmetic requires another "
                f"{type(self).__name__}"
            )
        return other

    def __add__(self, other: object) -> "RationalBernstein":
        """Return the exact pointwise sum."""
        other = self._check_other(other)
        denominator = self.denominator * other.denominator
        numerator = self.numerator * other.denominator
        numerator = numerator + other.numerator * self.denominator
        return _from_homogeneous(numerator.c, denominator.c)

    def __sub__(self, other: object) -> "RationalBernstein":
        """Return the exact pointwise difference."""
        other = self._check_other(other)
        denominator = self.denominator * other.denominator
        numerator = self.numerator * other.denominator
        numerator = numerator - other.numerator * self.denominator
        return _from_homogeneous(numerator.c, denominator.c)

    def __mul__(self, other: object) -> "RationalBernstein":
        """Return the exact pointwise product."""
        other = self._check_other(other)
        numerator = self.numerator * other.numerator
        denominator = self.denominator * other.denominator
        return _from_homogeneous(numerator.c, denominator.c)

    def deriv(self, m: int = 1) -> "RationalBernstein":
        """Return the exact ``m``-th derivative using the quotient rule."""
        if m < 0:
            raise ValueError("derivative order must be non-negative")
        result = self
        for _ in range(m):
            numerator = (
                result.numerator.deriv() * result.denominator
                - result.numerator * result.denominator.deriv()
            )
            denominator = result.denominator * result.denominator
            numerator_coefficients = _elevate(numerator.c, denominator.order)
            result = _from_homogeneous(numerator_coefficients, denominator.c)
        return result

    def split(
        self, t: jax.Array | float = jnp.array(0.5)
    ) -> tuple["RationalBernstein", "RationalBernstein"]:
        """Split and reparameterize the rational function at ``t``."""
        left, right = _split_homogeneous(self.h, t)
        return (
            _from_homogeneous(left[..., 0, :], left[..., 1, :]),
            _from_homogeneous(right[..., 0, :], right[..., 1, :]),
        )


@jax.custom_jvp
def _minimize(h: jax.Array, max_steps: int = 200, eps: float = 1e-6):
    """Approximate the minimum of one unbatched positive-weight curve."""
    max_cap = max_steps + 1
    degree = h.shape[-1] - 1
    values = h[0] / h[1]

    u_buffer = jnp.zeros(max_cap).at[0].set(0.0)
    v_buffer = jnp.zeros(max_cap).at[0].set(1.0)
    h_buffer = jnp.zeros((max_cap, 2, degree + 1)).at[0].set(h)
    lower_buffer = jnp.full(max_cap, jnp.inf).at[0].set(jnp.min(values))
    upper = jnp.minimum(values[0], values[-1])
    best_x = jnp.where(values[0] < values[-1], 0.0, 1.0)
    initial = (u_buffer, v_buffer, h_buffer, lower_buffer, upper, best_x, 0)

    def condition(state):
        _, _, _, lower, incumbent, _, step = state
        return (step < max_steps) & ((incumbent - jnp.min(lower)) > eps)

    def body(state):
        u_buffer, v_buffer, h_buffer, lower, incumbent, best_x, step = state
        index = jnp.argmin(lower)
        u = u_buffer[index]
        v = v_buffer[index]
        current = h_buffer[index]
        midpoint = 0.5 * (u + v)
        left, right = _split_homogeneous(
            current, jnp.array(0.5, dtype=current.dtype)
        )
        left_values = left[0] / left[1]
        right_values = right[0] / right[1]

        samples = jnp.array(
            [left_values[0], left_values[-1], right_values[-1]]
        )
        sample_points = jnp.array([u, midpoint, v])
        sample_index = jnp.argmin(samples)
        new_upper = jnp.minimum(incumbent, samples[sample_index])
        new_best_x = jnp.where(
            samples[sample_index] < incumbent,
            sample_points[sample_index],
            best_x,
        )

        next_index = step + 1
        u_buffer = u_buffer.at[index].set(u).at[next_index].set(midpoint)
        v_buffer = v_buffer.at[index].set(midpoint).at[next_index].set(v)
        h_buffer = h_buffer.at[index].set(left).at[next_index].set(right)
        lower = (
            lower.at[index]
            .set(jnp.min(left_values))
            .at[next_index]
            .set(jnp.min(right_values))
        )
        return (
            u_buffer,
            v_buffer,
            h_buffer,
            lower,
            new_upper,
            new_best_x,
            step + 1,
        )

    final = jax.lax.while_loop(condition, body, initial)
    return final[4], final[5]


@_minimize.defjvp
def _minimize_jvp(primals, tangents):
    h, max_steps, eps = primals
    tangent_h, _, _ = tangents
    primal = _minimize(h, max_steps=max_steps, eps=eps)
    x_star = primal[1]

    tangent_f = jax.jvp(
        lambda homogeneous: _evaluate_homogeneous(homogeneous, x_star),
        (h,),
        (tangent_h,),
    )[1]

    def first_derivative(homogeneous):
        return jax.grad(
            lambda parameter: _evaluate_homogeneous(homogeneous, parameter)
        )(x_star)

    tangent_first = jax.jvp(first_derivative, (h,), (tangent_h,))[1]
    second = jax.grad(
        lambda parameter: jax.grad(
            lambda inner: _evaluate_homogeneous(h, inner)
        )(parameter)
    )(x_star)
    tangent_x = jax.lax.cond(
        jnp.equal(x_star, 0.0) | jnp.equal(x_star, 1.0),
        lambda _: jnp.zeros_like(x_star),
        lambda _: -tangent_first / second,
        operand=None,
    )
    return primal, (tangent_f, tangent_x)
