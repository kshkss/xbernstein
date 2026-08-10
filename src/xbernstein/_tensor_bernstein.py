from typing import ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jss
from jaxtyping import Float


class _TensorBernstein(eqx.Module):
    """Shared implementation for tensor-product Bernstein polynomials."""

    c: Float[jax.Array, "..."]
    parameter_dimensions: ClassVar[int]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.c.shape[: -self.parameter_dimensions]

    @property
    def dtype(self) -> str:
        return str(self.c.dtype)

    def _axis_index(self, axis: int) -> int:
        if not 0 <= axis < self.parameter_dimensions:
            raise ValueError(
                f"axis must be in [0, {self.parameter_dimensions}), got {axis}"
            )
        return self.c.ndim - self.parameter_dimensions + axis

    def _elevate_axis(self, c: jax.Array, axis: int, target_degree: int) -> jax.Array:
        axis_index = c.ndim - self.parameter_dimensions + axis
        w = jnp.moveaxis(c, axis_index, -1)
        degree = w.shape[-1] - 1

        for new_degree in range(degree + 1, target_degree + 1):
            i = jnp.arange(1, new_degree, dtype=w.dtype)
            alpha = i / new_degree
            middle = alpha * w[..., :-1] + (1.0 - alpha) * w[..., 1:]
            w = jnp.concatenate([w[..., :1], middle, w[..., -1:]], axis=-1)

        return jnp.moveaxis(w, -1, axis_index)

    def _elevate(self, c: jax.Array, target_degrees: tuple[int, ...]) -> jax.Array:
        for axis, target_degree in enumerate(target_degrees):
            c = self._elevate_axis(c, axis, target_degree)
        return c

    def _broadcast_coefficients(
        self, c: jax.Array, batch_shape: tuple[int, ...], degree_shape: tuple[int, ...]
    ) -> jax.Array:
        return jnp.broadcast_to(c, batch_shape + degree_shape)

    def _aligned_coefficients(self, other: Self) -> tuple[jax.Array, jax.Array]:
        d = self.parameter_dimensions
        degrees = tuple(
            max(self.c.shape[-d + axis], other.c.shape[-d + axis]) - 1
            for axis in range(d)
        )
        batch_shape = jnp.broadcast_shapes(self.shape, other.shape)
        degree_shape = tuple(degree + 1 for degree in degrees)
        c1 = self._broadcast_coefficients(
            self._elevate(self.c, degrees), batch_shape, degree_shape
        )
        c2 = self._broadcast_coefficients(
            self._elevate(other.c, degrees), batch_shape, degree_shape
        )
        return c1, c2

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, type(self)):
            return NotImplemented
        c1, c2 = self._aligned_coefficients(other)
        return type(self)(c1 + c2)

    def __sub__(self, other: Self) -> Self:
        if not isinstance(other, type(self)):
            return NotImplemented
        c1, c2 = self._aligned_coefficients(other)
        return type(self)(c1 - c2)

    def __mul__(self, other: Self) -> Self:
        if not isinstance(other, type(self)):
            return NotImplemented

        d = self.parameter_dimensions
        batch_shape = jnp.broadcast_shapes(self.shape, other.shape)
        n_shape = self.c.shape[-d:]
        m_shape = other.c.shape[-d:]
        c1 = self._broadcast_coefficients(self.c, batch_shape, n_shape)
        c2 = self._broadcast_coefficients(other.c, batch_shape, m_shape)

        grid_shape = n_shape + m_shape
        scale = jnp.ones(grid_shape, dtype=c1.dtype)
        flat_indices = jnp.zeros(grid_shape, dtype=jnp.int32)
        output_shape = tuple(n + m - 1 for n, m in zip(n_shape, m_shape))
        strides = tuple(
            int(jnp.prod(jnp.asarray(output_shape[axis + 1 :]))) for axis in range(d)
        )

        for axis, (n_size, m_size, stride) in enumerate(zip(n_shape, m_shape, strides)):
            i = jnp.arange(n_size).reshape(
                (1,) * axis + (n_size,) + (1,) * (2 * d - axis - 1)
            )
            j = jnp.arange(m_size).reshape(
                (1,) * (d + axis) + (m_size,) + (1,) * (d - axis - 1)
            )
            n = n_size - 1
            m = m_size - 1
            scale = scale * jss.comb(n, i) * jss.comb(m, j) / jss.comb(n + m, i + j)
            flat_indices = flat_indices + (i + j) * stride

        products = (
            c1.reshape(batch_shape + n_shape + (1,) * d)
            * c2.reshape(batch_shape + (1,) * d + m_shape)
            * scale
        )
        result = jnp.zeros(batch_shape + output_shape, dtype=products.dtype).reshape(
            batch_shape + (-1,)
        )
        result = result.at[..., flat_indices.reshape(-1)].add(
            products.reshape(batch_shape + (-1,))
        )
        return type(self)(result.reshape(batch_shape + output_shape))

    def deriv(self, m: int = 1, axis: int = 0) -> Self:
        """Differentiate ``m`` times with respect to a parameter axis."""
        axis_index = self._axis_index(axis)
        w = jnp.moveaxis(self.c, axis_index, -1)
        degree = w.shape[-1] - 1

        if m > degree:
            w = jnp.zeros(w.shape[:-1] + (1,), dtype=w.dtype)
        else:
            for current_degree in range(degree, degree - m, -1):
                w = current_degree * (w[..., 1:] - w[..., :-1])

        return type(self)(jnp.moveaxis(w, -1, axis_index))

    def int(self, k: float = 0.0, axis: int = 0) -> Self:
        """Integrate with respect to a parameter axis using constant ``k``."""
        axis_index = self._axis_index(axis)
        w = jnp.moveaxis(self.c, axis_index, -1)
        degree = w.shape[-1] - 1
        constant = jnp.broadcast_to(jnp.asarray(k, dtype=w.dtype), w.shape[:-1])
        integrated = jnp.concatenate(
            [
                constant[..., None],
                constant[..., None] + jnp.cumsum(w, axis=-1) / (degree + 1),
            ],
            axis=-1,
        )
        return type(self)(jnp.moveaxis(integrated, -1, axis_index))

    def __call__(self, *ts: Float[jax.Array, "..."]) -> jax.Array:
        """Evaluate at one coordinate array for each parameter axis."""
        if len(ts) != self.parameter_dimensions:
            raise TypeError(
                f"expected {self.parameter_dimensions} parameter arrays, got {len(ts)}"
            )

        d = self.parameter_dimensions
        batch_dimensions = self.c.ndim - d
        parameters = tuple(jnp.asarray(t, dtype=self.c.dtype) for t in ts)
        point_shape = jnp.broadcast_shapes(*(t.shape for t in parameters))
        parameters = tuple(jnp.broadcast_to(t, point_shape) for t in parameters)
        w = self.c.reshape(self.shape + (1,) * len(point_shape) + self.c.shape[-d:])

        for axis in range(d - 1, -1, -1):
            t = parameters[axis].reshape(
                (1,) * batch_dimensions + point_shape + (1,) * (axis + 1)
            )
            for _ in range(w.shape[-1] - 1):
                w = (1.0 - t) * w[..., :-1] + t * w[..., 1:]
            w = w[..., 0]

        return w

    def split(
        self, t: Float[jax.Array, "..."] = jnp.array(0.5), axis: int = 0
    ) -> tuple[Self, Self]:
        """Split the polynomial at ``t`` along one parameter axis."""
        axis_index = self._axis_index(axis)
        w = jnp.moveaxis(self.c, axis_index, -1)
        t = jnp.asarray(t, dtype=w.dtype)
        t = jnp.broadcast_to(t, self.shape).reshape(
            self.shape + (1,) * self.parameter_dimensions
        )

        left = [w[..., 0]]
        right = [w[..., -1]]
        for _ in range(w.shape[-1] - 1):
            w = (1.0 - t) * w[..., :-1] + t * w[..., 1:]
            left.append(w[..., 0])
            right.append(w[..., -1])

        left_coefficients = jnp.moveaxis(jnp.stack(left, axis=-1), -1, axis_index)
        right_coefficients = jnp.moveaxis(
            jnp.stack(right[::-1], axis=-1), -1, axis_index
        )
        return type(self)(left_coefficients), type(self)(right_coefficients)
