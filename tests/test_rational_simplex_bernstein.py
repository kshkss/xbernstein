import operator
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import (
    Bernstein2DS,
    RationalBernstein,
    RationalBernstein2DS,
    RationalBernstein3DS,
    RationalBernstein4DS,
    maximize,
    minimize,
)


class RationalSimplexBernsteinTest(unittest.TestCase):
    cases = (
        (RationalBernstein2DS, 2, 6, Bernstein2DS),
        (RationalBernstein3DS, 3, 10, None),
        (RationalBernstein4DS, 4, 15, None),
    )

    def test_constructor_properties_and_broadcasting(self):
        for rational_type, dimensions, count, polynomial_type in self.cases:
            values = jnp.arange(count, dtype=jnp.float32).reshape(
                (1, count)
            )
            weights = jnp.arange(1, 2 * count + 1, dtype=jnp.float32).reshape(
                (2, count)
            )
            function = rational_type(values, weights)

            with self.subTest(rational_type=rational_type.__name__):
                self.assertEqual(function.shape, (2,))
                self.assertEqual(function.order, 2)
                self.assertEqual(function.h.shape, (2, 2, count))
                self.assertEqual(function.multi_indices.shape, (count, dimensions + 1))
                npt.assert_allclose(function.c, function.values)
                npt.assert_allclose(function.w, function.weights)
                npt.assert_allclose(
                    function.h[..., 0, :], function.values * function.weights
                )
                self.assertEqual(
                    function.numerator.c.shape, function.values.shape
                )
                if polynomial_type is not None:
                    self.assertIsInstance(function.numerator, polynomial_type)

    def test_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "packed coefficient axis"):
            RationalBernstein2DS(1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "same packed coefficient count"):
            RationalBernstein2DS(jnp.ones(3), jnp.ones(6))
        with self.assertRaisesRegex(ValueError, "packed coefficient count"):
            RationalBernstein2DS(jnp.ones(2), jnp.ones(2))
        with self.assertRaisesRegex(eqx.EquinoxRuntimeError, "strictly positive"):
            RationalBernstein2DS(
                jnp.ones(3), jnp.array([1.0, 0.0, 1.0])
            )

    def test_evaluation_matches_homogeneous_ratio_and_weight_scaling(self):
        for rational_type, dimensions, count, _ in self.cases:
            values = jnp.arange(count, dtype=jnp.float32) / count
            weights = jnp.linspace(0.5, 2.0, count)
            function = rational_type(values, weights)
            point = jnp.arange(1, dimensions + 2, dtype=jnp.float32)
            point /= jnp.sum(point)

            with self.subTest(rational_type=rational_type.__name__):
                expected = function.numerator(*point) / function.denominator(
                    *point
                )
                npt.assert_allclose(function(*point), expected)
                npt.assert_allclose(
                    rational_type(values, 7.0 * weights)(*point),
                    function(*point),
                    rtol=1e-5,
                )

    def test_arithmetic_and_derivatives(self):
        for rational_type, dimensions, count, _ in self.cases:
            values = jnp.arange(count, dtype=jnp.float32) / count
            weights = jnp.linspace(0.5, 2.0, count)
            left = rational_type(values, weights)
            right = rational_type(0.25 - values, weights[::-1])
            point = jnp.arange(1, dimensions + 2, dtype=jnp.float32)
            point /= jnp.sum(point)

            for operation in (operator.add, operator.sub, operator.mul):
                with self.subTest(
                    rational_type=rational_type.__name__,
                    operation=operation.__name__,
                ):
                    actual = operation(left, right)
                    npt.assert_allclose(
                        actual(*point),
                        operation(left(*point), right(*point)),
                        rtol=3e-5,
                        atol=1e-6,
                    )

            gradient = jax.grad(lambda coordinates: left(*coordinates))(point)
            for axis in range(dimensions + 1):
                npt.assert_allclose(
                    left.deriv(axis=axis)(*point),
                    gradient[axis],
                    rtol=3e-5,
                    atol=2e-6,
                )

    def test_weight_sensitivity_matches_weight_jacobian(self):
        for rational_type, dimensions, count, _ in self.cases:
            with self.subTest(rational_type=rational_type.__name__):
                values = jnp.arange(count, dtype=jnp.float32) / count
                weights = jnp.linspace(0.5, 2.0, count)
                coordinates = tuple(
                    jnp.full(4, 1.0 / (dimensions + 1))
                    for _ in range(dimensions + 1)
                )
                function = rational_type(values, weights)
                actual = function.weight_sensitivity()
                expected = jax.jacrev(
                    lambda current_weights: rational_type(values, current_weights)(
                        *coordinates
                    )
                )(weights)

                self.assertIsInstance(actual, rational_type)
                self.assertEqual(actual.shape, (count,))
                self.assertEqual(actual.order, 2 * function.order)
                npt.assert_allclose(
                    jnp.moveaxis(actual(*coordinates), 0, -1),
                    expected,
                    rtol=5e-5,
                    atol=4e-6,
                )
                npt.assert_allclose(
                    jnp.sum(weights[:, None] * actual(*coordinates), axis=0),
                    0.0,
                    atol=4e-6,
                )

    def test_segment_is_exact(self):
        for rational_type, dimensions, count, _ in self.cases:
            values = jnp.arange(count, dtype=jnp.float32)
            weights = jnp.linspace(0.5, 2.0, count)
            function = rational_type(values, weights)
            start = jnp.eye(dimensions + 1)[0]
            end = jnp.full(dimensions + 1, 1.0 / (dimensions + 1))
            segment = function.segment(start, end)
            parameters = jnp.linspace(0.0, 1.0, 7)
            points = (
                (1.0 - parameters[:, None]) * start
                + parameters[:, None] * end
            )

            with self.subTest(rational_type=rational_type.__name__):
                self.assertIsInstance(segment, RationalBernstein)
                npt.assert_allclose(
                    segment(parameters),
                    function(
                        *(points[:, axis] for axis in range(dimensions + 1))
                    ),
                    rtol=3e-5,
                )

    def test_jax_transformations(self):
        weights = jnp.linspace(0.5, 2.0, 6)
        point = jnp.array([0.2, 0.3, 0.5])

        def evaluate(values):
            return RationalBernstein2DS(values, weights)(*point)

        values = jnp.arange(6.0)
        eager = evaluate(values)
        npt.assert_allclose(jax.jit(evaluate)(values), eager)
        npt.assert_allclose(
            jax.vmap(evaluate)(jnp.stack((values, values + 1))),
            jnp.array([eager, eager + 1]),
            rtol=1e-5,
        )
        npt.assert_allclose(
            jax.jacfwd(evaluate)(values),
            jax.jacrev(evaluate)(values),
            rtol=1e-5,
        )

    def test_minimize_maximize_and_jvp(self):
        values = jnp.array([0.0, 1.0, 2.0])
        weights = jnp.array([1.0, 2.0, 1.0])
        function = RationalBernstein2DS(values, weights)
        minimum = minimize(function, max_steps=20)
        maximum = maximize(function, max_steps=20)

        npt.assert_allclose(minimum.f, 0.0)
        npt.assert_allclose(minimum.x, [1.0, 0.0, 0.0])
        npt.assert_allclose(maximum.f, 2.0)
        npt.assert_allclose(maximum.x, [0.0, 0.0, 1.0])

        _, tangent = jax.jvp(
            lambda controls: minimize(
                RationalBernstein2DS(controls, weights), max_steps=20
            ),
            (values,),
            (jnp.ones_like(values),),
        )
        npt.assert_allclose(tangent.f, 1.0)
        npt.assert_allclose(tangent.x, 0.0)

        batched = RationalBernstein2DS(
            jnp.stack((values, values + 1.0)), weights
        )
        batched_result = minimize(batched, max_steps=20)
        self.assertEqual(batched_result.f.shape, (2,))
        self.assertEqual(batched_result.x.shape, (2, 3))
        npt.assert_allclose(batched_result.f, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
