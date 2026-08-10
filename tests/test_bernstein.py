import math
import operator
import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import Bernstein, Bernstein2D
from xbernstein.bernstein import _elevate, minimize


def bernstein_basis(degree: int, t: jax.Array) -> jax.Array:
    indices = jnp.arange(degree + 1)
    return (
        jnp.asarray([math.comb(degree, int(index)) for index in indices])
        * t[..., None] ** indices
        * (1.0 - t[..., None]) ** (degree - indices)
    )


def assert_jax_transformations(
    test_case: unittest.TestCase, function, coefficients: jax.Array
) -> None:
    eager = function(coefficients)
    jitted = jax.jit(function)(coefficients)
    batched_coefficients = jnp.stack([coefficients, coefficients + 0.25])
    vmapped = jax.vmap(function)(batched_coefficients)
    expected_vmapped = jnp.stack([function(c) for c in batched_coefficients])
    forward_jacobian = jax.jacfwd(function)(coefficients)
    reverse_jacobian = jax.jacrev(function)(coefficients)

    npt.assert_allclose(jitted, eager, rtol=1e-5, atol=1e-6)
    npt.assert_allclose(vmapped, expected_vmapped, rtol=1e-5, atol=1e-6)
    test_case.assertEqual(forward_jacobian.shape, eager.shape + coefficients.shape)
    npt.assert_allclose(forward_jacobian, reverse_jacobian, rtol=1e-5, atol=1e-6)
    test_case.assertTrue(bool(jnp.all(jnp.isfinite(forward_jacobian))))


class BernsteinFormulaTest(unittest.TestCase):
    def test_arithmetic_broadcasts_multidimensional_batch_shapes(self):
        left = Bernstein(jnp.ones((2, 1, 3)))
        right = Bernstein(jnp.ones((1, 3, 2)))

        self.assertEqual((left + right).c.shape, (2, 3, 3))
        self.assertEqual((left - right).c.shape, (2, 3, 3))
        self.assertEqual((left * right).c.shape, (2, 3, 4))

    def test_arithmetic_rejects_tensor_polynomial_types(self):
        polynomial = Bernstein(jnp.ones(2))
        tensor_polynomial = Bernstein2D(jnp.ones((2, 2)))

        for operation in (operator.add, operator.sub, operator.mul):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(
                    TypeError, "Bernstein arithmetic requires another Bernstein"
                ):
                    operation(polynomial, tensor_polynomial)

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

    def test_antiderivative_broadcasts_batch_constants(self):
        polynomial = Bernstein(jnp.zeros((2, 3)))
        antiderivative = polynomial.int(k=jnp.array([1.0, 2.0]))

        self.assertEqual(antiderivative.c.shape, (2, 4))
        npt.assert_allclose(antiderivative(0.0), jnp.array([1.0, 2.0]))

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

    def test_public_operations_support_jax_transformations(self):
        other = jnp.array([-0.5, 1.5])
        operations = {
            "add": lambda c: (Bernstein(c) + Bernstein(other)).c,
            "sub": lambda c: (Bernstein(c) - Bernstein(other)).c,
            "mul": lambda c: (Bernstein(c) * Bernstein(other)).c,
            "deriv": lambda c: Bernstein(c).deriv(m=1).c,
            "int": lambda c: Bernstein(c).int(k=0.25).c,
            "call": lambda c: Bernstein(c)(jnp.array(0.4)),
            "split": lambda c: jnp.stack(
                tuple(piece.c for piece in Bernstein(c).split(jnp.array(0.4)))
            ),
        }
        coefficients = jnp.array([0.25, -0.5, 1.0])

        for name, operation in operations.items():
            with self.subTest(operation=name):
                assert_jax_transformations(self, operation, coefficients)

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

    def test_degree_zero_polynomial_operations(self):
        polynomial = Bernstein(jnp.array([2.0]))
        other = Bernstein(jnp.array([3.0]))
        parameters = jnp.linspace(0.0, 1.0, 5)
        left, right = polynomial.split(0.4)
        result = minimize(polynomial, max_steps=10, eps=1e-7)

        npt.assert_allclose(polynomial(parameters), jnp.full(5, 2.0))
        npt.assert_allclose((polynomial + other).c, jnp.array([5.0]))
        npt.assert_allclose(
            (polynomial * other).c, jnp.array([6.0]), rtol=1e-5, atol=1e-6
        )
        npt.assert_allclose(polynomial.deriv().c, jnp.array([0.0]))
        npt.assert_allclose(polynomial.int().c, jnp.array([0.0, 2.0]))
        npt.assert_allclose(left.c, polynomial.c)
        npt.assert_allclose(right.c, polynomial.c)
        npt.assert_allclose(result.f, 2.0)

    def test_minimize_multimodal_and_max_steps_limit(self):
        # (t - 0.2)^2 (t - 0.8)^2 in the degree-4 Bernstein basis.
        coefficients = jnp.array([0.0256, -0.0544, 0.0856, -0.0544, 0.0256])
        polynomial = Bernstein(coefficients)
        stopped = minimize(polynomial, max_steps=0, eps=1e-7)
        refined = minimize(polynomial, max_steps=100, eps=1e-7)

        npt.assert_allclose(stopped.f, 0.0256, atol=1e-7)
        npt.assert_allclose(stopped.x, 1.0, atol=1e-7)
        npt.assert_allclose(refined.f, 0.0, atol=1e-6)
        self.assertLess(
            min(abs(float(refined.x) - 0.2), abs(float(refined.x) - 0.8)), 1e-3
        )

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

    def test_minimize_jvp_composes_with_vmap(self):
        coefficients = jnp.array(
            [
                [0.09, -0.21, 0.49],
                [1.09, 0.79, 1.49],
            ]
        )
        tangents = jnp.array(
            [
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],
            ]
        )

        def minimum_value(coefficients):
            return minimize(Bernstein(coefficients), max_steps=100, eps=1e-7).f

        values, tangent_values = jax.vmap(
            lambda coefficients, tangent: jax.jvp(
                minimum_value, (coefficients,), (tangent,)
            )
        )(coefficients, tangents)

        self.assertEqual(values.shape, (2,))
        self.assertEqual(tangent_values.shape, (2,))
        npt.assert_allclose(values, jnp.array([0.0, 1.0]), atol=1e-6)
        npt.assert_allclose(tangent_values, jnp.array([0.1, 0.2]), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
