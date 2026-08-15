import itertools
import math
import operator
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from jaxtyping import TypeCheckError

from xbernstein import (
    Bernstein,
    Bernstein2DS,
    Bernstein3DS,
    Bernstein4DS,
    maximize,
    minimize,
)


def simplex_indices(dimensions, degree):
    return sorted(
        (
            alpha
            for alpha in itertools.product(
                range(degree + 1), repeat=dimensions + 1
            )
            if sum(alpha) == degree
        ),
        reverse=True,
    )


def multinomial(degree, alpha):
    result = math.factorial(degree)
    for value in alpha:
        result //= math.factorial(value)
    return result


def direct_value(coefficients, dimensions, point):
    degree = 0
    while math.comb(degree + dimensions, dimensions) < coefficients.shape[-1]:
        degree += 1
    return sum(
        coefficients[index]
        * multinomial(degree, alpha)
        * np.prod(np.asarray(point) ** alpha)
        for index, alpha in enumerate(simplex_indices(dimensions, degree))
    )


class SimplexBernsteinTest(unittest.TestCase):
    cases = (
        (Bernstein2DS, 2, 6),
        (Bernstein3DS, 3, 10),
        (Bernstein4DS, 4, 15),
    )

    def test_properties_and_packed_order(self):
        for polynomial_type, dimensions, count in self.cases:
            coefficients = jnp.zeros((2, 3, count))
            polynomial = polynomial_type(coefficients)

            with self.subTest(polynomial_type=polynomial_type.__name__):
                self.assertEqual(polynomial.shape, (2, 3))
                self.assertEqual(polynomial.order, 2)
                self.assertEqual(polynomial.dtype, "float32")
                npt.assert_array_equal(
                    polynomial.multi_indices,
                    simplex_indices(dimensions, 2),
                )

    def test_rejects_invalid_packed_counts(self):
        for polynomial_type, _, _ in self.cases:
            with self.subTest(polynomial_type=polynomial_type.__name__):
                with self.assertRaisesRegex(ValueError, "packed coefficient count"):
                    polynomial_type(jnp.ones(2))
                with self.assertRaisesRegex(TypeCheckError, "c"):
                    polynomial_type(1.0)

    def test_evaluation_matches_multinomial_basis_and_vertices(self):
        for polynomial_type, dimensions, count in self.cases:
            coefficients = jnp.arange(count, dtype=jnp.float32)
            polynomial = polynomial_type(coefficients)
            point = np.arange(1, dimensions + 2, dtype=float)
            point /= point.sum()

            with self.subTest(polynomial_type=polynomial_type.__name__):
                npt.assert_allclose(
                    polynomial(*point),
                    direct_value(np.asarray(coefficients), dimensions, point),
                    rtol=1e-6,
                )
                for vertex in range(dimensions + 1):
                    coordinates = np.eye(dimensions + 1)[vertex]
                    expected_index = simplex_indices(dimensions, 2).index(
                        tuple(2 if axis == vertex else 0 for axis in range(dimensions + 1))
                    )
                    npt.assert_allclose(
                        polynomial(*coordinates), coefficients[expected_index]
                    )

    def test_batch_and_coordinate_broadcasting(self):
        polynomial = Bernstein2DS(jnp.zeros((2, 3, 6)))
        actual = polynomial(
            jnp.zeros((4, 1)),
            jnp.zeros((1, 5)),
            jnp.ones((1, 5)),
        )
        self.assertEqual(actual.shape, (2, 3, 4, 5))

    def test_arithmetic_matches_pointwise_operations(self):
        for polynomial_type, dimensions, count in self.cases:
            left = polynomial_type(jnp.arange(count, dtype=jnp.float32) / count)
            right = polynomial_type(
                jnp.arange(dimensions + 1, dtype=jnp.float32)
            )
            point = jnp.arange(1, dimensions + 2, dtype=jnp.float32)
            point /= jnp.sum(point)

            for operation in (operator.add, operator.sub, operator.mul):
                with self.subTest(
                    polynomial_type=polynomial_type.__name__,
                    operation=operation.__name__,
                ):
                    actual = operation(left, right)
                    npt.assert_allclose(
                        actual(*point),
                        operation(left(*point), right(*point)),
                        rtol=2e-5,
                    )
            self.assertEqual((left + right).order, 2)
            self.assertEqual((left * right).order, 3)

    def test_arithmetic_broadcasts_only_batch_axes(self):
        left = Bernstein2DS(jnp.ones((2, 1, 6)))
        right = Bernstein2DS(jnp.ones((1, 3, 3)))
        self.assertEqual((left + right).c.shape, (2, 3, 6))
        self.assertEqual((left * right).c.shape, (2, 3, 10))
        with self.assertRaisesRegex(
            TypeError, "Bernstein2DS arithmetic requires another Bernstein2DS"
        ):
            _ = left + Bernstein3DS(jnp.ones(4))

    def test_barycentric_derivatives_match_automatic_differentiation(self):
        for polynomial_type, dimensions, count in self.cases:
            coefficients = jnp.arange(count, dtype=jnp.float32) / count
            polynomial = polynomial_type(coefficients)
            point = jnp.arange(1, dimensions + 2, dtype=jnp.float32)
            point /= jnp.sum(point)
            gradient = jax.grad(lambda x: polynomial(*x))(point)

            with self.subTest(polynomial_type=polynomial_type.__name__):
                for axis in range(dimensions + 1):
                    npt.assert_allclose(
                        polynomial.deriv(axis=axis)(*point),
                        gradient[axis],
                        rtol=2e-5,
                    )
                self.assertEqual(polynomial.deriv(m=3).order, 0)
                with self.assertRaisesRegex(ValueError, "axis must be"):
                    polynomial.deriv(axis=dimensions + 1)

    def test_segment_is_exact_and_batched(self):
        for polynomial_type, dimensions, count in self.cases:
            coefficients = jnp.arange(count, dtype=jnp.float32)
            polynomial = polynomial_type(coefficients)
            start = jnp.eye(dimensions + 1)[0]
            end = jnp.full(dimensions + 1, 1.0 / (dimensions + 1))
            segment = polynomial.segment(start, end)
            parameters = jnp.linspace(0.0, 1.0, 7)
            points = (
                (1.0 - parameters[:, None]) * start
                + parameters[:, None] * end
            )

            with self.subTest(polynomial_type=polynomial_type.__name__):
                self.assertIsInstance(segment, Bernstein)
                self.assertEqual(segment.order, polynomial.order)
                npt.assert_allclose(
                    segment(parameters),
                    polynomial(
                        *(points[:, axis] for axis in range(dimensions + 1))
                    ),
                    rtol=2e-5,
                )

        batched = Bernstein2DS(jnp.stack((jnp.arange(6.0), jnp.arange(6.0) + 1)))
        starts = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ends = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(batched.segment(starts, ends).c.shape, (2, 3))

    def test_public_operations_support_jax_transformations(self):
        coefficients = jnp.arange(6.0)
        point = jnp.array([0.2, 0.3, 0.5])

        def evaluate(c):
            return Bernstein2DS(c)(*point)

        eager = evaluate(coefficients)
        npt.assert_allclose(jax.jit(evaluate)(coefficients), eager)
        npt.assert_allclose(
            jax.vmap(evaluate)(jnp.stack((coefficients, coefficients + 1))),
            jnp.array([eager, eager + 1]),
        )
        npt.assert_allclose(
            jax.jacfwd(evaluate)(coefficients),
            jax.jacrev(evaluate)(coefficients),
            rtol=1e-5,
        )

    def test_minimize_maximize_and_jvp(self):
        for polynomial_type, dimensions, _ in self.cases:
            linear = polynomial_type(
                jnp.arange(dimensions + 1, dtype=jnp.float32)
            )
            minimum = minimize(linear, max_steps=10)
            maximum = maximize(linear, max_steps=10)

            with self.subTest(polynomial_type=polynomial_type.__name__):
                npt.assert_allclose(minimum.f, 0.0)
                npt.assert_allclose(minimum.x, jnp.eye(dimensions + 1)[0])
                npt.assert_allclose(maximum.f, dimensions)
                npt.assert_allclose(maximum.x, jnp.eye(dimensions + 1)[-1])
                self.assertEqual(minimum.x.shape, (dimensions + 1,))

        x = Bernstein2DS(jnp.array([1.0, 0.0, 0.0]))
        y = Bernstein2DS(jnp.array([0.0, 1.0, 0.0]))
        constant_x = Bernstein2DS(jnp.array([0.2]))
        constant_y = Bernstein2DS(jnp.array([0.3]))
        quadratic = (x - constant_x) * (x - constant_x) + (
            y - constant_y
        ) * (y - constant_y)
        result = minimize(quadratic, max_steps=200, eps=1e-6)
        npt.assert_allclose(result.f, 0.0, atol=1e-6)
        npt.assert_allclose(result.x, [0.2, 0.3, 0.5], atol=5e-4)

        _, tangent = jax.jvp(
            lambda c: minimize(Bernstein2DS(c), max_steps=200, eps=1e-6),
            (quadratic.c,),
            (jnp.ones_like(quadratic.c),),
        )
        npt.assert_allclose(tangent.f, 1.0, atol=1e-5)
        npt.assert_allclose(tangent.x, 0.0, atol=1e-5)

        batched = Bernstein2DS(
            jnp.stack((quadratic.c, quadratic.c + 1.0))
        )
        batched_result = minimize(batched, max_steps=200, eps=1e-6)
        self.assertEqual(batched_result.f.shape, (2,))
        self.assertEqual(batched_result.x.shape, (2, 3))
        npt.assert_allclose(batched_result.f, [0.0, 1.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
