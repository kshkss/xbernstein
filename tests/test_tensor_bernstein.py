import itertools
import math
import operator
import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import (
    Bernstein,
    Bernstein2D,
    Bernstein3D,
    Bernstein4D,
    maximize,
    minimize,
)


def tensor_value_from_basis(coefficients, parameters):
    degrees = tuple(size - 1 for size in coefficients.shape)
    total = 0.0
    for index in itertools.product(*(range(size) for size in coefficients.shape)):
        basis = 1.0
        for axis, (degree, coefficient_index, parameter) in enumerate(
            zip(degrees, index, parameters)
        ):
            basis *= (
                math.comb(degree, coefficient_index)
                * parameter**coefficient_index
                * (1.0 - parameter) ** (degree - coefficient_index)
            )
        total += coefficients[index] * basis
    return total


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


class TensorBernsteinTest(unittest.TestCase):
    cases = (
        (Bernstein2D, (2, 2), (0.25, 0.75)),
        (Bernstein3D, (2, 2, 2), (0.25, 0.5, 0.75)),
        (Bernstein4D, (2, 2, 2, 2), (0.25, 0.5, 0.75, 0.125)),
    )

    def test_arithmetic_broadcasts_multidimensional_batch_shapes(self):
        for polynomial_type, dimensions in (
            (Bernstein2D, 2),
            (Bernstein3D, 3),
            (Bernstein4D, 4),
        ):
            left = polynomial_type(jnp.ones((2, 1) + (2,) * dimensions))
            right = polynomial_type(jnp.ones((1, 3) + (3,) * dimensions))

            with self.subTest(polynomial_type=polynomial_type.__name__):
                self.assertEqual((left + right).c.shape, (2, 3) + (3,) * dimensions)
                self.assertEqual((left - right).c.shape, (2, 3) + (3,) * dimensions)
                self.assertEqual((left * right).c.shape, (2, 3) + (4,) * dimensions)

    def test_arithmetic_rejects_different_tensor_dimensions(self):
        polynomial_2d = Bernstein2D(jnp.ones((2, 2)))
        polynomial_3d = Bernstein3D(jnp.ones((2, 2, 2)))

        for operation in (operator.add, operator.sub, operator.mul):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(
                    TypeError, "Bernstein2D arithmetic requires another Bernstein2D"
                ):
                    operation(polynomial_2d, polynomial_3d)

    def test_order_matches_parameter_coefficient_axes(self):
        cases = (
            (Bernstein2D, (2, 3)),
            (Bernstein3D, (2, 3, 4)),
            (Bernstein4D, (2, 3, 4, 5)),
        )

        for polynomial_type, coefficient_shape in cases:
            polynomial = polynomial_type(jnp.zeros((7,) + coefficient_shape))
            order = polynomial.order

            self.assertEqual(order.shape, (len(coefficient_shape),))
            self.assertTrue(jnp.issubdtype(order.dtype, jnp.integer))
            npt.assert_array_equal(order, jnp.asarray(coefficient_shape) - 1)
            self.assertEqual(
                polynomial.c.shape[-len(coefficient_shape) :],
                tuple((order + 1).tolist()),
            )

    def test_evaluation_corners_and_jit(self):
        for polynomial_type, shape, parameters in self.cases:
            coefficients = jnp.arange(jnp.prod(jnp.asarray(shape)), dtype=jnp.float32)
            coefficients = coefficients.reshape(shape)
            polynomial = polynomial_type(coefficients)
            lower = polynomial(*(0.0,) * len(shape))
            upper = polynomial(*(1.0,) * len(shape))

            npt.assert_allclose(lower, coefficients[(0,) * len(shape)])
            npt.assert_allclose(upper, coefficients[(-1,) * len(shape)])
            npt.assert_allclose(
                jax.jit(lambda *args: polynomial(*args))(*parameters),
                polynomial(*parameters),
            )

    def test_arithmetic_and_axis_operations(self):
        for polynomial_type, shape, parameters in self.cases:
            coefficients = jnp.arange(jnp.prod(jnp.asarray(shape)), dtype=jnp.float32)
            polynomial = polynomial_type(coefficients.reshape(shape))
            constant = polynomial_type(jnp.ones((1,) * len(shape), dtype=jnp.float32))

            npt.assert_allclose(
                (polynomial + constant)(*parameters), polynomial(*parameters) + 1
            )
            npt.assert_allclose(
                (polynomial - constant)(*parameters), polynomial(*parameters) - 1
            )
            npt.assert_allclose(
                (polynomial * polynomial)(*parameters),
                polynomial(*parameters) ** 2,
                rtol=1e-5,
            )

            for axis in range(len(shape)):
                self.assertEqual(polynomial.deriv(axis=axis).c.shape[axis], 1)
                self.assertEqual(polynomial.int(axis=axis).c.shape[axis], 3)

                left, right = polynomial.split(0.5, axis=axis)
                split_parameters = list(parameters)
                split_parameters[axis] = 1.0
                npt.assert_allclose(
                    left(*split_parameters),
                    polynomial(*parameters[:axis], 0.5, *parameters[axis + 1 :]),
                )
                split_parameters[axis] = 0.0
                npt.assert_allclose(
                    right(*split_parameters),
                    polynomial(*parameters[:axis], 0.5, *parameters[axis + 1 :]),
                )

    def test_public_operations_support_jax_transformations(self):
        for polynomial_type, shape, _ in self.cases:
            dimensions = len(shape)
            other = jnp.ones(shape)
            start = jnp.zeros(dimensions)
            end = jnp.linspace(0.25, 0.75, dimensions)

            def split_coefficients(c):
                lower, upper = polynomial_type(c).split(jnp.array(0.4), axis=0)
                return jnp.stack((lower.c, upper.c))

            operations = {
                "add": lambda c: (polynomial_type(c) + polynomial_type(other)).c,
                "sub": lambda c: (polynomial_type(c) - polynomial_type(other)).c,
                "mul": lambda c: (polynomial_type(c) * polynomial_type(other)).c,
                "deriv": lambda c: polynomial_type(c).deriv(m=1, axis=0).c,
                "int": lambda c: polynomial_type(c).int(k=0.25, axis=0).c,
                "call": lambda c: polynomial_type(c)(*(0.4,) * dimensions),
                "split": split_coefficients,
                "slice": lambda c: polynomial_type(c).slice(0.4, axis=0).c,
                "integrate_out": lambda c: polynomial_type(c).integrate_out(axis=0).c,
                "segment": lambda c: polynomial_type(c).segment(start, end).c,
            }
            coefficients = jnp.arange(math.prod(shape), dtype=jnp.float32).reshape(
                shape
            )

            for name, operation in operations.items():
                with self.subTest(
                    polynomial_type=polynomial_type.__name__, operation=name
                ):
                    assert_jax_transformations(self, operation, coefficients)

    def test_batched_evaluation(self):
        polynomial = Bernstein2D(
            jnp.array(
                [
                    [[0.0, 1.0], [2.0, 3.0]],
                    [[4.0, 5.0], [6.0, 7.0]],
                ]
            )
        )
        npt.assert_allclose(polynomial(0.25, 0.75), jnp.array([1.25, 5.25]))

    def test_minimize_tensor_polynomials_and_jvp(self):
        for polynomial_type, dimensions in (
            (Bernstein2D, 2),
            (Bernstein3D, 3),
            (Bernstein4D, 4),
        ):
            coefficients = jnp.zeros((2,) * dimensions).at[(1,) * dimensions].set(-1.0)
            polynomial = polynomial_type(coefficients)
            result = minimize(polynomial, max_steps=20, eps=1e-7)
            batched = polynomial_type(jnp.stack([coefficients, coefficients + 2.0]))

            with self.subTest(polynomial_type=polynomial_type.__name__):
                npt.assert_allclose(result.f, -1.0)
                npt.assert_allclose(result.x, jnp.ones(dimensions))
                self.assertEqual(minimize(batched, max_steps=20).f.shape, (2,))
                self.assertEqual(
                    minimize(batched, max_steps=20).x.shape, (2, dimensions)
                )

                def minimum(coefficients):
                    return minimize(polynomial_type(coefficients), max_steps=20).f

                _, tangent = jax.jvp(
                    minimum, (coefficients,), (jnp.ones_like(coefficients),)
                )
                npt.assert_allclose(tangent, 1.0)
                npt.assert_allclose(
                    jax.vmap(minimum)(jnp.stack([coefficients, coefficients + 2.0])),
                    jnp.array([-1.0, 1.0]),
                )

    def test_maximize_tensor_polynomials_and_jvp(self):
        for polynomial_type, dimensions in (
            (Bernstein2D, 2),
            (Bernstein3D, 3),
            (Bernstein4D, 4),
        ):
            coefficients = jnp.zeros((2,) * dimensions).at[(1,) * dimensions].set(1.0)

            def maximum(c):
                return maximize(polynomial_type(c), max_steps=20).f

            with self.subTest(polynomial_type=polynomial_type.__name__):
                result = maximize(polynomial_type(coefficients), max_steps=20)
                npt.assert_allclose(result.f, 1.0)
                npt.assert_allclose(result.x, jnp.ones(dimensions))
                npt.assert_allclose(
                    result.f, -minimize(polynomial_type(-coefficients), max_steps=20).f
                )
                self.assertEqual(
                    maximize(
                        polynomial_type(jnp.stack([coefficients, coefficients - 2.0])),
                        max_steps=20,
                    ).x.shape,
                    (2, dimensions),
                )
                _, tangent = jax.jvp(
                    maximum, (coefficients,), (jnp.ones_like(coefficients),)
                )
                npt.assert_allclose(tangent, 1.0)
                npt.assert_allclose(
                    jax.vmap(maximum)(jnp.stack([coefficients, coefficients - 2.0])),
                    jnp.array([1.0, -1.0]),
                )

    def test_segment_matches_tensor_evaluation(self):
        cases = (
            (Bernstein2D, (3, 2), jnp.array([0.1, 0.2]), jnp.array([0.8, 0.9])),
            (
                Bernstein3D,
                (2, 3, 2),
                jnp.array([0.1, 0.2, 0.3]),
                jnp.array([0.8, 0.7, 0.9]),
            ),
            (
                Bernstein4D,
                (2, 2, 3, 2),
                jnp.array([0.1, 0.2, 0.3, 0.4]),
                jnp.array([0.8, 0.7, 0.9, 0.6]),
            ),
        )
        parameters = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])

        for polynomial_type, shape, start, end in cases:
            coefficients = jnp.arange(jnp.prod(jnp.asarray(shape)), dtype=jnp.float32)
            polynomial = polynomial_type(coefficients.reshape(shape))
            restricted = polynomial.segment(start, end)
            coordinates = start[None, :] + parameters[:, None] * (end - start)

            npt.assert_allclose(
                restricted(parameters),
                polynomial(*(coordinates[:, axis] for axis in range(len(shape)))),
                rtol=1e-5,
            )
            npt.assert_allclose(
                jax.jit(lambda s, e, t: polynomial.segment(s, e)(t))(
                    start, end, parameters
                ),
                restricted(parameters),
                rtol=1e-5,
            )

    def test_segment_supports_degenerate_and_batched_endpoints(self):
        polynomial = Bernstein2D(
            jnp.array(
                [
                    [[0.0, 1.0], [2.0, 3.0]],
                    [[4.0, 5.0], [6.0, 7.0]],
                ]
            )
        )
        start = jnp.array([[0.25, 0.75], [0.5, 0.25]])
        end = jnp.array([[0.25, 0.75], [0.75, 0.5]])
        parameters = jnp.array([0.0, 0.5, 1.0])
        restricted = polynomial.segment(start, end)

        coordinates = (
            start[:, None, :] + parameters[None, :, None] * (end - start)[:, None, :]
        )
        expected = jnp.stack(
            [
                Bernstein2D(polynomial.c[index])(
                    coordinates[index, :, 0], coordinates[index, :, 1]
                )
                for index in range(2)
            ]
        )
        npt.assert_allclose(restricted(parameters), expected, rtol=1e-5)

    def test_segment_rejects_invalid_endpoint_shapes(self):
        for polynomial_type, dimensions in (
            (Bernstein2D, 2),
            (Bernstein3D, 3),
            (Bernstein4D, 4),
        ):
            polynomial = polynomial_type(jnp.ones((1,) * dimensions))

            with self.subTest(polynomial_type=polynomial_type.__name__, case="scalar"):
                with self.assertRaises(ValueError):
                    polynomial.segment(jnp.array(0.0), jnp.zeros(dimensions))
            with self.subTest(
                polynomial_type=polynomial_type.__name__, case="coordinate_length"
            ):
                with self.assertRaises(ValueError):
                    polynomial.segment(jnp.zeros(dimensions - 1), jnp.zeros(dimensions))
            with self.subTest(
                polynomial_type=polynomial_type.__name__, case="batch_shape"
            ):
                with self.assertRaises(ValueError):
                    polynomial.segment(
                        jnp.zeros((2, dimensions)), jnp.zeros((3, dimensions))
                    )

    def test_slice_and_integrate_out_reduce_dimensions(self):
        cases = (
            (Bernstein2D, Bernstein, (2, 3), (0.25, 0.75)),
            (Bernstein3D, Bernstein2D, (2, 3, 2), (0.25, 0.5, 0.75)),
            (Bernstein4D, Bernstein3D, (2, 2, 3, 2), (0.25, 0.5, 0.75, 0.125)),
        )

        for polynomial_type, reduced_type, shape, parameters in cases:
            coefficients = jnp.arange(jnp.prod(jnp.asarray(shape)), dtype=jnp.float32)
            polynomial = polynomial_type(coefficients.reshape(shape))

            for axis, value in enumerate(parameters):
                sliced = polynomial.slice(value, axis=axis)
                integrated = polynomial.integrate_out(axis=axis)
                remaining_parameters = parameters[:axis] + parameters[axis + 1 :]
                axis_index = polynomial.c.ndim - len(shape) + axis

                self.assertIsInstance(sliced, reduced_type)
                self.assertIsInstance(integrated, reduced_type)
                self.assertEqual(
                    sliced.c.shape,
                    tuple(size for index, size in enumerate(shape) if index != axis),
                )
                npt.assert_allclose(
                    sliced(*remaining_parameters), polynomial(*parameters)
                )
                npt.assert_allclose(
                    integrated.c,
                    jnp.sum(polynomial.c, axis=axis_index) / shape[axis],
                )

                antiderivative = polynomial.int(axis=axis)
                lower_coordinates = list(remaining_parameters)
                upper_coordinates = list(remaining_parameters)
                lower_coordinates.insert(axis, 0.0)
                upper_coordinates.insert(axis, 1.0)
                npt.assert_allclose(
                    integrated(*remaining_parameters),
                    antiderivative(*upper_coordinates)
                    - antiderivative(*lower_coordinates),
                )
                npt.assert_allclose(
                    jax.jit(
                        lambda fixed_value: polynomial.slice(fixed_value, axis=axis)(
                            *remaining_parameters
                        )
                    )(value),
                    polynomial(*parameters),
                )

    def test_degree_zero_tensor_polynomial_operations(self):
        for polynomial_type, dimensions in (
            (Bernstein2D, 2),
            (Bernstein3D, 3),
            (Bernstein4D, 4),
        ):
            polynomial = polynomial_type(jnp.full((1,) * dimensions, 2.0))
            parameters = (0.25,) * dimensions
            lower, upper = polynomial.split(axis=0)
            sliced = polynomial.slice(0.25, axis=0)
            integrated = polynomial.integrate_out(axis=0)
            segment = polynomial.segment(jnp.zeros(dimensions), jnp.ones(dimensions))

            with self.subTest(polynomial_type=polynomial_type.__name__):
                npt.assert_allclose(polynomial(*parameters), 2.0)
                npt.assert_allclose(
                    polynomial.deriv(axis=0).c, jnp.zeros_like(polynomial.c)
                )
                npt.assert_allclose(lower.c, polynomial.c)
                npt.assert_allclose(upper.c, polynomial.c)
                npt.assert_allclose(sliced.c, jnp.full((1,) * (dimensions - 1), 2.0))
                npt.assert_allclose(
                    integrated.c, jnp.full((1,) * (dimensions - 1), 2.0)
                )
                npt.assert_allclose(segment.c, jnp.array([2.0]))

    def test_slice_supports_batched_values_and_validates_axes(self):
        polynomial = Bernstein2D(
            jnp.array(
                [
                    [[0.0, 1.0], [2.0, 3.0]],
                    [[4.0, 5.0], [6.0, 7.0]],
                ]
            )
        )
        values = jnp.array([0.25, 0.75])
        sliced = polynomial.slice(values, axis=0)
        expected = jnp.array(
            [
                Bernstein2D(polynomial.c[index])(values[index], 0.5)
                for index in range(values.shape[0])
            ]
        )

        npt.assert_allclose(sliced(0.5), expected)
        with self.assertRaises(ValueError):
            polynomial.slice(0.5, axis=2)
        with self.assertRaises(ValueError):
            polynomial.integrate_out(axis=-1)

    def test_tensor_basis_elevation_and_calculus_coefficient_formulas(self):
        cases = (
            (Bernstein2D, (2, 3), (0.25, 0.75)),
            (Bernstein3D, (2, 3, 2), (0.25, 0.5, 0.75)),
            (Bernstein4D, (2, 2, 3, 2), (0.25, 0.5, 0.75, 0.125)),
        )

        for polynomial_type, shape, parameters in cases:
            coefficients = jnp.arange(jnp.prod(jnp.asarray(shape)), dtype=jnp.float32)
            coefficients = coefficients.reshape(shape)
            polynomial = polynomial_type(coefficients)
            npt.assert_allclose(
                polynomial(*parameters),
                tensor_value_from_basis(coefficients, parameters),
                rtol=1e-5,
            )

            for axis, axis_size in enumerate(shape):
                moved = jnp.moveaxis(coefficients, axis, -1)
                degree = axis_size - 1
                elevated_moved = jnp.concatenate(
                    [
                        moved[..., :1],
                        jnp.arange(1, axis_size, dtype=coefficients.dtype)
                        / axis_size
                        * moved[..., :-1]
                        + (
                            1.0
                            - jnp.arange(1, axis_size, dtype=coefficients.dtype)
                            / axis_size
                        )
                        * moved[..., 1:],
                        moved[..., -1:],
                    ],
                    axis=-1,
                )
                elevated = polynomial._elevate_axis(coefficients, axis, degree + 1)
                npt.assert_allclose(
                    elevated,
                    jnp.moveaxis(elevated_moved, -1, axis),
                    rtol=1e-5,
                )
                npt.assert_allclose(
                    polynomial_type(elevated)(*parameters),
                    polynomial(*parameters),
                    rtol=1e-5,
                )

                expected_derivative = degree * (moved[..., 1:] - moved[..., :-1])
                npt.assert_allclose(
                    polynomial.deriv(axis=axis).c,
                    jnp.moveaxis(expected_derivative, -1, axis),
                )

                expected_integral = jnp.concatenate(
                    [
                        jnp.zeros(moved.shape[:-1] + (1,)),
                        jnp.cumsum(moved, axis=-1) / axis_size,
                    ],
                    axis=-1,
                )
                npt.assert_allclose(
                    polynomial.int(axis=axis).c,
                    jnp.moveaxis(expected_integral, -1, axis),
                )

    def test_tensor_product_multi_index_coefficient_formula(self):
        first_coefficients = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        second_coefficients = jnp.arange(6, 12, dtype=jnp.float32).reshape(3, 2)
        first = Bernstein2D(first_coefficients)
        second = Bernstein2D(second_coefficients)
        expected = jnp.zeros((4, 4), dtype=jnp.float32)

        for first_index in itertools.product(
            *(range(size) for size in first_coefficients.shape)
        ):
            for second_index in itertools.product(
                *(range(size) for size in second_coefficients.shape)
            ):
                output_index = tuple(i + j for i, j in zip(first_index, second_index))
                scale = math.prod(
                    math.comb(first_coefficients.shape[axis] - 1, first_index[axis])
                    * math.comb(second_coefficients.shape[axis] - 1, second_index[axis])
                    / math.comb(
                        first_coefficients.shape[axis]
                        + second_coefficients.shape[axis]
                        - 2,
                        output_index[axis],
                    )
                    for axis in range(2)
                )
                expected = expected.at[output_index].add(
                    first_coefficients[first_index]
                    * second_coefficients[second_index]
                    * scale
                )

        product = first * second
        npt.assert_allclose(product.c, expected, rtol=1e-5)
        npt.assert_allclose(
            product(0.25, 0.75),
            first(0.25, 0.75) * second(0.25, 0.75),
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
