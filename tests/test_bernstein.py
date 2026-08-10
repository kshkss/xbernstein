import math
import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import Bernstein
from xbernstein.bernstein import _elevate, minimize


def bernstein_basis(degree: int, t: jax.Array) -> jax.Array:
    indices = jnp.arange(degree + 1)
    return (
        jnp.asarray([math.comb(degree, int(index)) for index in indices])
        * t[..., None] ** indices
        * (1.0 - t[..., None]) ** (degree - indices)
    )


class BernsteinFormulaTest(unittest.TestCase):
    def test_order_matches_final_coefficient_axis(self):
        scalar = Bernstein(jnp.array([1.0, 2.0, 3.0, 4.0]))
        batched = Bernstein(jnp.zeros((2, 3, 5)))

        self.assertEqual(scalar.order, 3)
        self.assertEqual(batched.order, 4)
        self.assertEqual(scalar.c.shape[-1], scalar.order + 1)
        self.assertEqual(batched.c.shape[-1], batched.order + 1)

    def test_degree_elevation_recurrence_and_evaluation(self):
        coefficients = jnp.array([2.0, -1.0, 3.0])
        elevated = _elevate(coefficients, 3)
        expected = jnp.array(
            [
                coefficients[0],
                coefficients[0] / 3.0 + 2.0 * coefficients[1] / 3.0,
                2.0 * coefficients[1] / 3.0 + coefficients[2] / 3.0,
                coefficients[2],
            ]
        )
        parameters = jnp.linspace(0.0, 1.0, 9)

        npt.assert_allclose(elevated, expected, rtol=1e-5, atol=1e-6)
        npt.assert_allclose(
            Bernstein(elevated)(parameters),
            Bernstein(coefficients)(parameters),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_arithmetic_and_product_coefficient_formula(self):
        c = jnp.array([1.0, -2.0, 3.0])
        d = jnp.array([2.0, 4.0])
        p = Bernstein(c)
        q = Bernstein(d)
        expected_product = jnp.zeros(4)

        for i in range(c.shape[0]):
            for j in range(d.shape[0]):
                scale = math.comb(2, i) * math.comb(1, j) / math.comb(3, i + j)
                expected_product = expected_product.at[i + j].add(c[i] * d[j] * scale)

        parameters = jnp.linspace(0.0, 1.0, 7)
        npt.assert_allclose(
            (p + q)(parameters), p(parameters) + q(parameters), rtol=1e-5
        )
        npt.assert_allclose(
            (p - q)(parameters), p(parameters) - q(parameters), rtol=1e-5
        )
        npt.assert_allclose((p * q).c, expected_product, rtol=1e-5)
        npt.assert_allclose(
            (p * q)(parameters), p(parameters) * q(parameters), rtol=1e-5, atol=1e-6
        )

    def test_calculus_coefficient_formulas_and_boundary_condition(self):
        coefficients = jnp.array([1.0, -2.0, 3.0, 5.0])
        polynomial = Bernstein(coefficients)
        derivative = polynomial.deriv()
        antiderivative = polynomial.int(k=2.5)
        expected_derivative = 3.0 * (coefficients[1:] - coefficients[:-1])
        expected_antiderivative = jnp.array(
            [2.5, 2.5 + 1.0 / 4.0, 2.5 - 1.0 / 4.0, 2.5 + 2.0 / 4.0, 2.5 + 7.0 / 4.0]
        )
        parameters = jnp.linspace(0.0, 1.0, 11)

        npt.assert_allclose(derivative.c, expected_derivative)
        npt.assert_allclose(antiderivative.c, expected_antiderivative)
        npt.assert_allclose(antiderivative(0.0), 2.5)
        npt.assert_allclose(antiderivative.deriv()(parameters), polynomial(parameters))
        npt.assert_allclose(polynomial.deriv(m=5).c, jnp.zeros(1))

    def test_basis_sum_and_de_casteljau_evaluation(self):
        coefficients = jnp.array([1.0, -2.0, 3.0, 5.0])
        polynomial = Bernstein(coefficients)
        parameters = jnp.array([0.0, 0.2, 0.5, 0.8, 1.0])
        expected = bernstein_basis(3, parameters) @ coefficients

        npt.assert_allclose(polynomial(parameters), expected, rtol=1e-5)

    def test_split_reparameterization(self):
        polynomial = Bernstein(jnp.array([1.0, -2.0, 3.0, 5.0]))
        split_at = 0.35
        left, right = polynomial.split(split_at)
        parameters = jnp.linspace(0.0, 1.0, 9)

        npt.assert_allclose(
            left(parameters), polynomial(split_at * parameters), rtol=1e-5
        )
        npt.assert_allclose(
            right(parameters),
            polynomial(split_at + (1.0 - split_at) * parameters),
            rtol=1e-5,
        )

    def test_minimize_and_batch_shape(self):
        # (t - 0.3)^2 in the degree-2 Bernstein basis.
        interior = Bernstein(jnp.array([0.09, -0.21, 0.49]))
        boundary = Bernstein(jnp.array([0.0, 1.0]))
        interior_result = minimize(interior, max_steps=100, eps=1e-7)
        boundary_result = minimize(boundary, max_steps=100, eps=1e-7)
        batched = Bernstein(jnp.stack([interior.c, interior.c + 1.0]))
        batched_result = minimize(batched, max_steps=100, eps=1e-7)

        npt.assert_allclose(interior_result.f, 0.0, atol=1e-6)
        npt.assert_allclose(interior_result.x, 0.3, atol=1e-3)
        npt.assert_allclose(boundary_result.f, 0.0, atol=1e-7)
        npt.assert_allclose(boundary_result.x, 0.0, atol=1e-7)
        self.assertEqual(interior.order, 2)
        self.assertEqual(boundary.order, 1)
        self.assertEqual(batched_result.shape, (2,))
        npt.assert_allclose(batched_result.f, jnp.array([0.0, 1.0]), atol=1e-6)

    def test_minimize_jvp_matches_documented_rules(self):
        coefficients = jnp.array([0.09, -0.21, 0.49])
        tangent = jnp.array([0.5, -0.25, 0.75])

        def minimum(coefficients):
            result = minimize(Bernstein(coefficients), max_steps=100, eps=1e-7)
            return result.f, result.x

        (minimum_value, minimizer), (tangent_value, tangent_minimizer) = jax.jvp(
            minimum, (coefficients,), (tangent,)
        )
        polynomial = Bernstein(coefficients)
        tangent_polynomial = Bernstein(tangent)
        expected_value = tangent_polynomial(minimizer)
        expected_minimizer = -tangent_polynomial.deriv()(minimizer) / polynomial.deriv(
            m=2
        )(minimizer)

        npt.assert_allclose(minimum_value, 0.0, atol=1e-6)
        npt.assert_allclose(tangent_value, expected_value, atol=1e-6)
        npt.assert_allclose(tangent_minimizer, expected_minimizer, atol=1e-5)

        _, (_, boundary_tangent_minimizer) = jax.jvp(
            minimum, (jnp.array([0.0, 1.0]),), (jnp.array([1.0, -1.0]),)
        )
        npt.assert_allclose(boundary_tangent_minimizer, 0.0)


if __name__ == "__main__":
    unittest.main()
