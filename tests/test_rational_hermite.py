import itertools
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from xbernstein import (
    RationalBernstein,
    RationalBernstein2D,
    RationalBernstein3D,
    RationalBernstein4D,
    rational_hermite_interpolate_1d,
    rational_hermite_interpolate_2d,
    rational_hermite_interpolate_3d,
    rational_hermite_interpolate_4d,
)
from xbernstein.rational_hermite import _axis_subsets, _subset_indices


def _multi_indices(dimensions, total):
    return sorted(
        (
            alpha
            for alpha in itertools.product(range(3), repeat=dimensions)
            if sum(alpha) == total
        ),
        reverse=True,
    )


def _polynomial_vertex_jet(coefficients, alpha, vertex):
    result = coefficients
    for axis, order in enumerate(alpha):
        if order:
            result = np.diff(result, n=order, axis=axis)
            result *= math.factorial(3) / math.factorial(3 - order)
    endpoint = tuple(0 if coordinate == 0 else -1 for coordinate in vertex)
    return result[endpoint]


def _endpoint_data(values, weights):
    dimensions = values.ndim
    vertices = tuple(itertools.product((0, 1), repeat=dimensions))
    numerator = values * weights
    jets = {}

    for total in range(2 * dimensions + 1):
        for alpha in _multi_indices(dimensions, total):
            vertex_jets = {}
            for vertex in vertices:
                value = _polynomial_vertex_jet(numerator, alpha, vertex)
                gamma_ranges = (range(order + 1) for order in alpha)
                for gamma in itertools.product(*gamma_ranges):
                    if gamma == alpha:
                        continue
                    coefficient = math.prod(
                        math.comb(alpha[axis], gamma[axis])
                        for axis in range(dimensions)
                    )
                    denominator_alpha = tuple(
                        alpha[axis] - gamma[axis] for axis in range(dimensions)
                    )
                    value -= (
                        coefficient
                        * jets[gamma][vertex]
                        * _polynomial_vertex_jet(weights, denominator_alpha, vertex)
                    )
                vertex_jets[vertex] = value / _polynomial_vertex_jet(
                    weights, (0,) * dimensions, vertex
                )
            jets[alpha] = vertex_jets

    groups = []
    for total in range(2 * dimensions + 1):
        values_in_group = []
        for alpha in _multi_indices(dimensions, total):
            values_in_group.append(
                np.asarray([jets[alpha][vertex] for vertex in vertices])
                .reshape((2,) * dimensions)
                .astype(np.float32)
            )
        groups.append(
            values_in_group[0]
            if len(values_in_group) == 1
            else np.stack(values_in_group, axis=-1)
        )
    return groups, jets


def _source(dimensions, seed=0):
    rng = np.random.default_rng(seed)
    coefficient_shape = (4,) * dimensions
    values = rng.normal(scale=0.3, size=coefficient_shape)
    weights = rng.uniform(0.5, 1.5, size=coefficient_shape)
    indices = np.indices(coefficient_shape)
    corners = np.ones(coefficient_shape, dtype=bool)
    for axis in range(dimensions):
        corners &= (indices[axis] == 0) | (indices[axis] == 3)
    return values, np.where(corners, 1.0, weights)


class RationalHermiteInterpolationTest(unittest.TestCase):
    cases = (
        (rational_hermite_interpolate_1d, RationalBernstein, 1),
        (rational_hermite_interpolate_2d, RationalBernstein2D, 2),
        (rational_hermite_interpolate_3d, RationalBernstein3D, 3),
        (rational_hermite_interpolate_4d, RationalBernstein4D, 4),
    )

    def test_axis_subsets_are_solved_as_disjoint_systems(self):
        dimensions = 3
        vertices = tuple(itertools.product((0, 1), repeat=dimensions))
        square_free = tuple(itertools.product((0, 1), repeat=dimensions))
        derivative_jet_indices = tuple(
            index
            for index, (_, beta) in enumerate(itertools.product(vertices, square_free))
            if any(beta)
        )

        self.assertEqual(
            tuple(_axis_subsets(dimensions)),
            ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)),
        )
        rows_x, columns_x = _subset_indices(
            dimensions, (0,), derivative_jet_indices
        )
        rows_y, columns_y = _subset_indices(
            dimensions, (1,), derivative_jet_indices
        )
        rows_xy, columns_xy = _subset_indices(
            dimensions, (0, 1), derivative_jet_indices
        )

        self.assertEqual(rows_x.size, 2 * len(vertices))
        self.assertEqual(columns_x.size, 2 * len(vertices))
        self.assertFalse(bool(jnp.any(jnp.isin(columns_x, columns_y))))
        self.assertFalse(bool(jnp.any(jnp.isin(columns_x, columns_xy))))
        self.assertFalse(bool(jnp.any(jnp.isin(rows_x, rows_y))))

    def test_recovers_known_cubic_rational_functions(self):
        for interpolate, rational_type, dimensions in self.cases:
            with self.subTest(interpolate=interpolate.__name__):
                values, weights = _source(dimensions)
                groups, _ = _endpoint_data(values, weights)
                actual = interpolate(*(jnp.asarray(group) for group in groups))

                self.assertIsInstance(actual, rational_type)
                npt.assert_array_equal(
                    jnp.atleast_1d(actual.order), jnp.full(dimensions, 3)
                )
                tolerance = 7e-3 if dimensions == 4 else 5e-4
                npt.assert_allclose(actual.weights, weights, atol=tolerance)
                npt.assert_allclose(
                    actual.values * actual.weights,
                    values * weights,
                    atol=tolerance,
                )

    def test_reproduces_nonzero_vertex_values(self):
        values = jnp.array([2.0, 5.0])
        d1 = jnp.array([3.0, 3.0])
        d2 = jnp.zeros(2)

        actual = rational_hermite_interpolate_1d(values, d1, d2)

        npt.assert_allclose(actual(0.0), values[0], atol=2e-5)
        npt.assert_allclose(actual(1.0), values[1], atol=2e-5)

    def test_reproduces_selected_vertex_derivatives(self):
        values, weights = _source(2)
        groups, expected_jets = _endpoint_data(values, weights)
        actual = rational_hermite_interpolate_2d(
            *(jnp.asarray(group) for group in groups)
        )
        _, actual_jets = _endpoint_data(
            np.asarray(actual.values), np.asarray(actual.weights)
        )
        selected = set(itertools.product((0, 1), repeat=2))
        selected.update(itertools.product((0, 2), repeat=2))
        selected.remove((0, 0))

        for alpha in selected:
            for vertex in itertools.product((0, 1), repeat=2):
                self.assertAlmostEqual(
                    actual_jets[alpha][vertex],
                    expected_jets[alpha][vertex],
                    delta=3e-3,
                )

    def test_unreproduced_derivatives_affect_weights(self):
        values, weights = _source(2)
        groups, _ = _endpoint_data(values, weights)
        original = rational_hermite_interpolate_2d(
            *(jnp.asarray(group) for group in groups)
        )
        changed_groups = list(groups)
        changed_groups[3] = changed_groups[3].copy()
        changed_groups[3][0, 0, 0] += 0.1
        changed = rational_hermite_interpolate_2d(
            *(jnp.asarray(group) for group in changed_groups)
        )

        self.assertGreater(
            float(jnp.max(jnp.abs(changed.weights - original.weights))), 1e-5
        )

    def test_preserves_batch_axes(self):
        sources = [_source(2, seed) for seed in (0, 1)]
        data = [_endpoint_data(values, weights)[0] for values, weights in sources]
        groups = [
            jnp.asarray(np.stack([source_groups[index] for source_groups in data]))
            for index in range(5)
        ]

        actual = rational_hermite_interpolate_2d(*groups)

        self.assertEqual(actual.shape, (2,))
        for batch, (values, weights) in enumerate(sources):
            npt.assert_allclose(actual.weights[batch], weights, atol=5e-4)
            npt.assert_allclose(
                actual.values[batch] * actual.weights[batch],
                values * weights,
                atol=5e-4,
            )

    def test_is_jittable_and_differentiable(self):
        f = jnp.array([0.0, 1.0])
        d1 = jnp.ones(2)
        d2 = jnp.zeros(2)
        weights = jax.jit(
            lambda value, first, second: (
                rational_hermite_interpolate_1d(value, first, second).weights
            )
        )(f, d1, d2)
        gradient = jax.grad(
            lambda value: rational_hermite_interpolate_1d(value, d1, d2)(0.4)
        )(f)

        self.assertEqual(weights.shape, (4,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))

    def test_singular_system_uses_stage_minimum_norm_solution(self):
        actual = rational_hermite_interpolate_1d(
            jnp.array([0.0, 1.0]), jnp.ones(2), jnp.zeros(2)
        )

        self.assertTrue(bool(jnp.all(actual.weights > 0.0)))
        npt.assert_allclose(actual(jnp.array([0.2, 0.7])), [0.2, 0.7], atol=2e-6)

    def test_constant_singular_system_uses_stage_minimum_norm_solution(self):
        actual = rational_hermite_interpolate_1d(
            jnp.array([2.0, 2.0]), jnp.zeros(2), jnp.zeros(2)
        )

        self.assertTrue(bool(jnp.all(actual.weights > 0.0)))
        npt.assert_allclose(actual(jnp.array([0.2, 0.7])), [2.0, 2.0], atol=2e-6)

    def test_rejects_non_positive_weights(self):
        with self.assertRaisesRegex(Exception, "non-positive weights"):
            rational_hermite_interpolate_1d(
                jnp.array([2.040919, -2.555665]),
                jnp.array([0.418099, -0.567770]),
                jnp.array([-0.452649, -0.215597]),
            )

    def test_rejects_invalid_group_shape(self):
        with self.assertRaisesRegex(ValueError, "derivative group 1 must have shape"):
            rational_hermite_interpolate_2d(
                jnp.zeros((2, 2)),
                jnp.zeros((2, 2, 3)),
                jnp.zeros((2, 2, 3)),
                jnp.zeros((2, 2, 2)),
                jnp.zeros((2, 2)),
            )


if __name__ == "__main__":
    unittest.main()
