import itertools
import unittest

import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import (
    Bernstein,
    Bernstein2D,
    Bernstein3D,
    Bernstein4D,
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


def multi_indices(dimensions: int, limit: int, total: int):
    return sorted(
        (
            alpha
            for alpha in itertools.product(range(limit), repeat=dimensions)
            if sum(alpha) == total
        ),
        reverse=True,
    )


def derivative(polynomial, alpha, dimensions: int):
    result = polynomial
    for axis, order in enumerate(alpha):
        if order == 0:
            continue
        if dimensions == 1:
            result = result.deriv(m=order)
        else:
            result = result.deriv(m=order, axis=axis)
    return result


def vertex_values(polynomial, dimensions: int):
    values = []
    for vertex in itertools.product((0.0, 1.0), repeat=dimensions):
        if dimensions == 1:
            values.append(polynomial(vertex[0]))
        else:
            values.append(polynomial(*vertex))
    return jnp.stack(values, axis=-1).reshape(
        polynomial.shape + (2,) * dimensions
    )


def endpoint_groups(polynomial, dimensions: int, degree: int):
    limit = (degree + 1) // 2
    groups = [vertex_values(polynomial, dimensions)]
    for total in range(1, (limit - 1) * dimensions + 1):
        values = [
            vertex_values(
                derivative(polynomial, alpha, dimensions),
                dimensions,
            )
            for alpha in multi_indices(dimensions, limit, total)
        ]
        groups.append(values[0] if len(values) == 1 else jnp.stack(values, axis=-1))
    return groups


def coefficients(dimensions: int, degree: int):
    coefficient_shape = (degree + 1,) * dimensions
    size = (degree + 1) ** dimensions
    base = jnp.arange(size, dtype=jnp.float32).reshape(coefficient_shape) / size
    return jnp.stack((base, 0.25 - 0.5 * base))


class InterpolationTest(unittest.TestCase):
    cases = (
        (linear_interpolate_1d, Bernstein, 1, 1),
        (linear_interpolate_2d, Bernstein2D, 2, 1),
        (linear_interpolate_3d, Bernstein3D, 3, 1),
        (linear_interpolate_4d, Bernstein4D, 4, 1),
        (hermite_interpolate_1d, Bernstein, 1, 3),
        (hermite_interpolate_2d, Bernstein2D, 2, 3),
        (hermite_interpolate_3d, Bernstein3D, 3, 3),
        (hermite_interpolate_4d, Bernstein4D, 4, 3),
        (quintic_hermite_interpolate_1d, Bernstein, 1, 5),
        (quintic_hermite_interpolate_2d, Bernstein2D, 2, 5),
        (quintic_hermite_interpolate_3d, Bernstein3D, 3, 5),
        (quintic_hermite_interpolate_4d, Bernstein4D, 4, 5),
    )

    def test_reconstructs_batched_coefficients(self):
        for interpolate, polynomial_type, dimensions, degree in self.cases:
            with self.subTest(interpolate=interpolate.__name__):
                expected = coefficients(dimensions, degree)
                source = polynomial_type(expected)
                actual = interpolate(*endpoint_groups(source, dimensions, degree))

                self.assertIsInstance(actual, polynomial_type)
                npt.assert_array_equal(
                    jnp.atleast_1d(actual.order), jnp.full(dimensions, degree)
                )
                self.assertEqual(actual.shape, source.shape)
                npt.assert_allclose(actual.c, expected, rtol=2e-5, atol=2e-5)

    def test_rejects_invalid_value_group_shape(self):
        f = jnp.zeros((2, 2))
        d1 = jnp.zeros((2, 2, 2, 3))
        d2 = jnp.zeros((2, 2, 2, 3))
        d3 = jnp.zeros((2, 2, 2))

        with self.assertRaisesRegex(
            ValueError, "value group must end with one length-2 axis per dimension"
        ):
            hermite_interpolate_3d(f, d1, d2, d3)

    def test_rejects_invalid_derivative_group_shape(self):
        f = jnp.zeros((2, 2, 2))
        d1 = jnp.zeros((2, 2, 2, 2))
        d2 = jnp.zeros((2, 2, 2, 3))
        d3 = jnp.zeros((2, 2, 2))

        with self.assertRaisesRegex(
            ValueError,
            r"derivative group 1 must have shape \(2, 2, 2, 3\)",
        ):
            hermite_interpolate_3d(f, d1, d2, d3)


if __name__ == "__main__":
    unittest.main()
