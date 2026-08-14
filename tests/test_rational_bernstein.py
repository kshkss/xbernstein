import math
import operator
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import (
    Bernstein,
    RationalBernstein,
    RationalBernstein2D,
    RationalBernstein3D,
    RationalBernstein4D,
    maximize,
    minimize,
)


def bernstein_basis(degree: int, t: jax.Array) -> jax.Array:
    indices = jnp.arange(degree + 1)
    return (
        jnp.asarray([math.comb(degree, int(index)) for index in indices])
        * t[..., None] ** indices
        * (1.0 - t[..., None]) ** (degree - indices)
    )


class RationalBernsteinTest(unittest.TestCase):
    def test_constructor_broadcasts_and_exposes_properties(self):
        values = jnp.arange(6.0).reshape(2, 1, 3)
        weights = jnp.arange(1.0, 13.0).reshape(1, 4, 3)
        curve = RationalBernstein(values, weights)
        expected_values = jnp.broadcast_to(values, (2, 4, 3))
        expected_weights = jnp.broadcast_to(weights, (2, 4, 3))

        self.assertEqual(curve.h.shape, (2, 4, 2, 3))
        self.assertEqual(curve.shape, (2, 4))
        self.assertEqual(curve.order, 2)
        self.assertEqual(curve.dtype, "float32")
        npt.assert_allclose(curve.values, expected_values)
        npt.assert_allclose(curve.weights, expected_weights)
        npt.assert_allclose(curve.c, expected_values)
        npt.assert_allclose(curve.w, expected_weights)
        npt.assert_allclose(curve.h[..., 0, :], expected_values * expected_weights)
        npt.assert_allclose(curve.h[..., 1, :], expected_weights)
        npt.assert_allclose(curve.numerator.c, expected_values * expected_weights)
        npt.assert_allclose(curve.denominator.c, expected_weights)

    def test_constructor_rejects_invalid_shapes_and_weights(self):
        with self.assertRaisesRegex(ValueError, "must have a coefficient axis"):
            RationalBernstein(1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "same coefficient-axis length"):
            RationalBernstein(jnp.ones(2), jnp.ones(3))
        with self.assertRaisesRegex(eqx.EquinoxRuntimeError, "strictly positive"):
            RationalBernstein(jnp.ones(2), jnp.array([1.0, 0.0]))
        with self.assertRaisesRegex(jax.errors.JaxRuntimeError, "strictly positive"):
            jax.jit(lambda w: RationalBernstein(jnp.ones(2), w).h)(
                jnp.array([1.0, -1.0])
            )

    def test_evaluation_matches_rational_basis_formula_and_shapes(self):
        values = jnp.array([0.25, -0.5, 1.5])
        weights = jnp.array([1.0, 3.0, 2.0])
        curve = RationalBernstein(values, weights)
        parameters = jnp.linspace(0.0, 1.0, 7)
        basis = bernstein_basis(2, parameters)
        expected = (basis @ (weights * values)) / (basis @ weights)

        npt.assert_allclose(curve(parameters), expected, rtol=1e-5)
        npt.assert_allclose(curve(0.0), values[0])
        npt.assert_allclose(curve(1.0), values[-1])

        batched = RationalBernstein(jnp.stack((values, values + 1.0)), weights)
        self.assertEqual(batched(jnp.zeros((4, 1))).shape, (2, 4, 1))

    def test_weight_scaling_preserves_function(self):
        values = jnp.array([0.0, 1.0, -0.5])
        weights = jnp.array([1.0, 2.0, 4.0])
        parameters = jnp.linspace(0.0, 1.0, 9)

        npt.assert_allclose(
            RationalBernstein(values, weights)(parameters),
            RationalBernstein(values, 7.0 * weights)(parameters),
            rtol=1e-5,
        )

    def test_arithmetic_matches_pointwise_operations_and_broadcasts(self):
        left = RationalBernstein(
            jnp.array([[0.0, 1.0, -0.5], [1.0, 0.5, 2.0]]),
            jnp.array([1.0, 2.0, 1.0]),
        )
        right = RationalBernstein(
            jnp.array([0.25, -1.0]),
            jnp.array([3.0, 1.0]),
        )
        parameters = jnp.linspace(0.0, 1.0, 7)

        for operation, expected_operation in (
            (operator.add, operator.add),
            (operator.sub, operator.sub),
            (operator.mul, operator.mul),
        ):
            with self.subTest(operation=operation.__name__):
                actual = operation(left, right)
                expected = expected_operation(left(parameters), right(parameters))
                self.assertEqual(actual.order, left.order + right.order)
                self.assertEqual(actual.shape, (2,))
                npt.assert_allclose(actual(parameters), expected, rtol=2e-5, atol=1e-6)

        for operation in (operator.add, operator.sub, operator.mul):
            with self.subTest(mixed_operation=operation.__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    "RationalBernstein arithmetic requires another RationalBernstein",
                ):
                    operation(left, Bernstein(jnp.ones(2)))

    def test_derivative_matches_automatic_differentiation(self):
        curve = RationalBernstein(
            jnp.array([0.25, -0.5, 1.5]),
            jnp.array([1.0, 3.0, 2.0]),
        )
        parameters = jnp.linspace(0.05, 0.95, 7)
        expected_first = jax.vmap(jax.grad(curve))(parameters)
        expected_second = jax.vmap(jax.grad(jax.grad(curve)))(parameters)

        npt.assert_allclose(curve.deriv()(parameters), expected_first, rtol=2e-5)
        npt.assert_allclose(
            curve.deriv(m=2)(parameters),
            expected_second,
            rtol=2e-4,
            atol=2e-5,
        )
        self.assertIs(curve.deriv(m=0), curve)
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            curve.deriv(m=-1)

    def test_weight_sensitivity_matches_weight_jacobian(self):
        values = jnp.array([0.25, -0.5, 1.5])
        weights = jnp.array([1.0, 3.0, 2.0])
        parameters = jnp.linspace(0.1, 0.9, 5)
        curve = RationalBernstein(values, weights)

        actual = curve.weight_sensitivity()
        expected = jax.jacrev(
            lambda current_weights: RationalBernstein(values, current_weights)(
                parameters
            )
        )(weights)

        self.assertIsInstance(actual, RationalBernstein)
        self.assertEqual(actual.shape, (3,))
        self.assertEqual(actual.order, 4)
        npt.assert_allclose(
            jnp.moveaxis(actual(parameters), 0, -1), expected, rtol=3e-5, atol=2e-6
        )
        npt.assert_allclose(
            jnp.sum(weights[:, None] * actual(parameters), axis=0), 0.0, atol=2e-6
        )

    def test_weight_sensitivity_preserves_batch_axes(self):
        values = jnp.array([[0.0, 1.0, -0.5], [1.0, 0.5, 2.0]])
        weights = jnp.array([1.0, 2.0, 4.0])
        sensitivity = RationalBernstein(values, weights).weight_sensitivity()

        self.assertEqual(sensitivity.shape, (2, 3))
        self.assertEqual(sensitivity(jnp.array([0.2, 0.7])).shape, (2, 3, 2))

    def test_split_reparameterizes_exactly(self):
        curve = RationalBernstein(
            jnp.array([[0.0, 1.0, -0.5], [1.0, 0.5, 2.0]]),
            jnp.array([1.0, 2.0, 1.0]),
        )
        split_at = jnp.array([0.3, 0.7])
        left, right = curve.split(split_at)
        parameters = jnp.linspace(0.0, 1.0, 7)
        curves = [
            RationalBernstein(curve.values[index], curve.weights[index])
            for index in range(2)
        ]
        expected_left = jnp.stack(
            [item(split_at[index] * parameters) for index, item in enumerate(curves)]
        )
        expected_right = jnp.stack(
            [
                item(split_at[index] + (1.0 - split_at[index]) * parameters)
                for index, item in enumerate(curves)
            ]
        )

        npt.assert_allclose(
            left(parameters),
            expected_left,
            rtol=2e-5,
        )
        npt.assert_allclose(
            right(parameters),
            expected_right,
            rtol=2e-5,
        )

    def test_public_operations_support_jax_transformations(self):
        weights = jnp.array([1.0, 2.0, 1.5])
        parameters = jnp.array([0.2, 0.6])

        def evaluate(values):
            return RationalBernstein(values, weights)(parameters)

        values = jnp.array([0.25, -0.5, 1.0])
        eager = evaluate(values)
        npt.assert_allclose(jax.jit(evaluate)(values), eager)
        npt.assert_allclose(
            jax.vmap(evaluate)(jnp.stack((values, values + 1.0))),
            jnp.stack((eager, eager + 1.0)),
            rtol=1e-5,
        )
        npt.assert_allclose(
            jax.jacfwd(evaluate)(values),
            jax.jacrev(evaluate)(values),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_degree_zero_operations(self):
        curve = RationalBernstein(jnp.array([2.0]), jnp.array([3.0]))
        parameters = jnp.linspace(0.0, 1.0, 5)
        left, right = curve.split(0.4)

        npt.assert_allclose(curve(parameters), jnp.full(5, 2.0))
        npt.assert_allclose(curve.deriv().values, jnp.array([0.0]))
        npt.assert_allclose(left(parameters), curve(parameters))
        npt.assert_allclose(right(parameters), curve(parameters))

    def test_minimize_maximize_batch_and_jvp(self):
        weights = jnp.array([1.0, 2.0, 1.0])
        numerator = jnp.array([0.09, -0.21, 0.49])
        values = numerator / weights
        curve = RationalBernstein(values, weights)
        result = minimize(curve, max_steps=100, eps=1e-7)
        maximum = maximize(
            RationalBernstein(jnp.array([0.0, 1.0]), jnp.array([2.0, 1.0])),
            max_steps=20,
            eps=1e-7,
        )

        npt.assert_allclose(result.f, 0.0, atol=1e-6)
        npt.assert_allclose(result.x, 0.3, atol=1e-3)
        npt.assert_allclose(maximum.f, 1.0, atol=1e-6)
        npt.assert_allclose(maximum.x, 1.0, atol=1e-6)

        batched = RationalBernstein(
            jnp.stack((values, values + 1.0)),
            weights,
        )
        batched_result = minimize(batched, max_steps=100, eps=1e-7)
        self.assertEqual(batched_result.shape, (2,))
        npt.assert_allclose(batched_result.f, jnp.array([0.0, 1.0]), atol=1e-6)

        def minimum(control_values):
            return minimize(
                RationalBernstein(control_values, weights),
                max_steps=100,
                eps=1e-7,
            )

        tangent = jnp.ones_like(values)
        _, tangent_result = jax.jvp(minimum, (values,), (tangent,))
        npt.assert_allclose(tangent_result.f, 1.0, atol=1e-5)
        npt.assert_allclose(tangent_result.x, 0.0, atol=1e-5)

        tangent = jnp.array([0.2, -0.4, 0.7])
        _, tangent_result = jax.jvp(minimum, (values,), (tangent,))
        tangent_curve = RationalBernstein(tangent, weights)
        expected_f = tangent_curve(result.x)
        expected_x = -tangent_curve.deriv()(result.x) / curve.deriv(m=2)(result.x)
        npt.assert_allclose(tangent_result.f, expected_f, rtol=1e-5, atol=1e-6)
        npt.assert_allclose(tangent_result.x, expected_x, rtol=1e-4, atol=1e-5)


class RationalTensorBernsteinTest(unittest.TestCase):
    cases = (
        (RationalBernstein2D, 2, RationalBernstein),
        (RationalBernstein3D, 3, RationalBernstein2D),
        (RationalBernstein4D, 4, RationalBernstein3D),
    )

    @staticmethod
    def controls(dimensions: int):
        shape = (2,) * dimensions
        indices = jnp.indices(shape)
        values = jnp.sum(indices, axis=0, dtype=jnp.float32)
        weights = 1.0 + jnp.arange(math.prod(shape), dtype=jnp.float32).reshape(shape)
        return values, weights

    def test_constructor_properties_evaluation_and_batching(self):
        for rational_type, dimensions, _ in self.cases:
            with self.subTest(rational_type=rational_type.__name__):
                values, weights = self.controls(dimensions)
                batched_values = jnp.stack((values, values + 1.0))[:, None, ...]
                batched_weights = jnp.stack((weights, 2.0 * weights))[None, ...]
                function = rational_type(batched_values, batched_weights)
                parameters = (0.2,) * dimensions

                self.assertEqual(function.shape, (2, 2))
                self.assertEqual(function.h.shape, (2, 2, 2) + (2,) * dimensions)
                npt.assert_array_equal(function.order, jnp.ones(dimensions))
                npt.assert_allclose(
                    function.values,
                    jnp.broadcast_to(batched_values, (2, 2) + (2,) * dimensions),
                )
                npt.assert_allclose(
                    function.weights,
                    jnp.broadcast_to(batched_weights, (2, 2) + (2,) * dimensions),
                )
                npt.assert_allclose(
                    function(*parameters),
                    function.numerator(*parameters) / function.denominator(*parameters),
                )
                point_arrays = (jnp.zeros((3, 1)), jnp.zeros((1, 4)))
                evaluation_parameters = point_arrays + (0.25,) * (dimensions - 2)
                self.assertEqual(
                    function(*evaluation_parameters).shape,
                    (2, 2, 3, 4),
                )

    def test_constructor_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "2 coefficient axes"):
            RationalBernstein2D(jnp.ones(2), jnp.ones(2))
        with self.assertRaisesRegex(ValueError, "same coefficient-axis shape"):
            RationalBernstein2D(jnp.ones((2, 2)), jnp.ones((2, 3)))
        with self.assertRaisesRegex(eqx.EquinoxRuntimeError, "strictly positive"):
            RationalBernstein2D(jnp.ones((2, 2)), jnp.array([[1.0, 1.0], [0.0, 1.0]]))

    def test_arithmetic_and_partial_derivatives(self):
        for rational_type, dimensions, _ in self.cases:
            with self.subTest(rational_type=rational_type.__name__):
                values, weights = self.controls(dimensions)
                left = rational_type(values, weights)
                right = rational_type(0.25 - values, weights[::-1])
                parameters = (0.2,) * dimensions

                for operation in (operator.add, operator.sub, operator.mul):
                    actual = operation(left, right)
                    expected = operation(left(*parameters), right(*parameters))
                    npt.assert_allclose(
                        actual(*parameters), expected, rtol=2e-5, atol=1e-6
                    )
                    npt.assert_array_equal(actual.order, left.order + right.order)

                point = jnp.full(dimensions, 0.3)
                expected_gradient = jax.grad(lambda coordinates: left(*coordinates))(
                    point
                )
                for axis in (0, 1):
                    npt.assert_allclose(
                        left.deriv(axis=axis)(*point),
                        expected_gradient[axis],
                        rtol=2e-5,
                        atol=1e-6,
                    )

    def test_weight_sensitivity_matches_weight_jacobian(self):
        for rational_type, dimensions, _ in self.cases:
            with self.subTest(rational_type=rational_type.__name__):
                values, weights = self.controls(dimensions)
                parameters = tuple(jnp.linspace(0.2, 0.7, 4) for _ in range(dimensions))
                function = rational_type(values, weights)
                actual = function.weight_sensitivity()
                expected = jax.jacrev(
                    lambda current_weights: rational_type(values, current_weights)(
                        *parameters
                    )
                )(weights)
                coefficient_axes = tuple(range(dimensions))
                target_axes = tuple(
                    range(actual(*parameters).ndim - dimensions, actual(*parameters).ndim)
                )

                self.assertIsInstance(actual, rational_type)
                npt.assert_array_equal(actual.order, 2 * function.order)
                self.assertEqual(actual.shape, weights.shape)
                npt.assert_allclose(
                    jnp.moveaxis(actual(*parameters), coefficient_axes, target_axes),
                    expected,
                    rtol=4e-5,
                    atol=3e-6,
                )
                npt.assert_allclose(
                    jnp.sum(
                        weights.reshape(weights.shape + (1,)) * actual(*parameters),
                        axis=coefficient_axes,
                    ),
                    0.0,
                    atol=3e-6,
                )

    def test_split_slice_and_segment(self):
        for rational_type, dimensions, lower_type in self.cases:
            with self.subTest(rational_type=rational_type.__name__):
                values, weights = self.controls(dimensions)
                function = rational_type(values, weights)
                split_at = 0.35
                parameters = [0.2] * dimensions
                left, right = function.split(split_at, axis=1)

                left_parameters = list(parameters)
                left_parameters[1] *= split_at
                right_parameters = list(parameters)
                right_parameters[1] = split_at + (1.0 - split_at) * parameters[1]
                npt.assert_allclose(
                    left(*parameters), function(*left_parameters), rtol=2e-5
                )
                npt.assert_allclose(
                    right(*parameters), function(*right_parameters), rtol=2e-5
                )

                sliced = function.slice(0.4, axis=1)
                sliced_parameters = parameters[:1] + parameters[2:]
                original_parameters = parameters.copy()
                original_parameters[1] = 0.4
                self.assertIsInstance(sliced, lower_type)
                npt.assert_allclose(
                    sliced(*sliced_parameters),
                    function(*original_parameters),
                    rtol=2e-5,
                )

                start = jnp.linspace(0.1, 0.2, dimensions)
                end = jnp.linspace(0.7, 0.9, dimensions)
                segment = function.segment(start, end)
                segment_parameter = 0.3
                coordinates = start + segment_parameter * (end - start)
                self.assertIsInstance(segment, RationalBernstein)
                npt.assert_allclose(
                    segment(segment_parameter),
                    function(*coordinates),
                    rtol=2e-5,
                )
                npt.assert_allclose(
                    jax.jit(lambda a, b, t: function.segment(a, b)(t))(
                        start, end, segment_parameter
                    ),
                    function(*coordinates),
                    rtol=2e-5,
                )

    def test_evaluation_supports_jax_transformations(self):
        for rational_type, dimensions, _ in self.cases:
            with self.subTest(rational_type=rational_type.__name__):
                values, weights = self.controls(dimensions)

                def evaluate(control_values):
                    return rational_type(control_values, weights)(*(0.3,) * dimensions)

                eager = evaluate(values)
                npt.assert_allclose(
                    jax.jit(evaluate)(values), eager, rtol=1e-5, atol=1e-6
                )
                npt.assert_allclose(
                    jax.vmap(evaluate)(jnp.stack((values, values + 1.0))),
                    jnp.array([eager, eager + 1.0]),
                    rtol=1e-5,
                )
                npt.assert_allclose(
                    jax.jacfwd(evaluate)(values),
                    jax.jacrev(evaluate)(values),
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_minimize_and_maximize(self):
        for rational_type, dimensions, _ in self.cases:
            with self.subTest(rational_type=rational_type.__name__):
                values, weights = self.controls(dimensions)
                function = rational_type(values, weights)
                minimum = minimize(function, max_steps=30, eps=1e-7)
                maximum = maximize(function, max_steps=30, eps=1e-7)

                npt.assert_allclose(minimum.f, 0.0, atol=1e-6)
                npt.assert_allclose(minimum.x, jnp.zeros(dimensions), atol=1e-6)
                npt.assert_allclose(maximum.f, float(dimensions), atol=1e-6)
                npt.assert_allclose(maximum.x, jnp.ones(dimensions), atol=1e-6)

                batched = rational_type(jnp.stack((values, values + 1.0)), weights)
                batched_minimum = minimize(batched, max_steps=30, eps=1e-7)
                self.assertEqual(batched_minimum.f.shape, (2,))
                self.assertEqual(batched_minimum.x.shape, (2, dimensions))
                npt.assert_allclose(batched_minimum.f, jnp.array([0.0, 1.0]), atol=1e-6)

    def test_minimize_supports_jvp(self):
        x_coefficients = jnp.array([0.09, -0.21, 0.49])
        y_coefficients = jnp.array([0.16, -0.24, 0.36])
        values = x_coefficients[:, None] + y_coefficients[None, :]
        weights = jnp.ones_like(values)

        def minimum(control_values):
            return minimize(
                RationalBernstein2D(control_values, weights),
                max_steps=100,
                eps=1e-7,
            )

        result, tangent = jax.jvp(
            minimum,
            (values,),
            (jnp.ones_like(values),),
        )
        npt.assert_allclose(result.f, 0.0, atol=5e-6)
        npt.assert_allclose(result.x, jnp.array([0.3, 0.4]), atol=1e-3)
        npt.assert_allclose(
            result.f,
            RationalBernstein2D(values, weights)(*result.x),
            atol=1e-7,
        )
        npt.assert_allclose(tangent.f, 1.0, atol=1e-5)
        npt.assert_allclose(tangent.x, jnp.zeros(2), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
