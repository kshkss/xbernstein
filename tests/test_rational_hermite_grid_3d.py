import itertools
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from xbernstein import (
    GridSegment3D,
    RationalBernstein3D,
    RationalHermiteGrid3D,
)


def _multi_indices(total):
    return sorted(
        (
            alpha
            for alpha in itertools.product(range(3), repeat=3)
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


def _endpoint_groups(values, weights):
    vertices = tuple(itertools.product((0, 1), repeat=3))
    numerator = values * weights
    jets = {}
    for total in range(7):
        for alpha in _multi_indices(total):
            vertex_jets = {}
            for vertex in vertices:
                value = _polynomial_vertex_jet(numerator, alpha, vertex)
                for gamma in itertools.product(
                    *(range(order + 1) for order in alpha)
                ):
                    if gamma == alpha:
                        continue
                    coefficient = math.prod(
                        math.comb(alpha[axis], gamma[axis]) for axis in range(3)
                    )
                    denominator_alpha = tuple(
                        alpha[axis] - gamma[axis] for axis in range(3)
                    )
                    value -= (
                        coefficient
                        * jets[gamma][vertex]
                        * _polynomial_vertex_jet(
                            weights, denominator_alpha, vertex
                        )
                    )
                vertex_jets[vertex] = value / _polynomial_vertex_jet(
                    weights, (0, 0, 0), vertex
                )
            jets[alpha] = vertex_jets

    groups = []
    for total in range(7):
        derivatives = [
            np.asarray([jets[alpha][vertex] for vertex in vertices]).reshape(
                (2, 2, 2)
            )
            for alpha in _multi_indices(total)
        ]
        groups.append(
            derivatives[0]
            if len(derivatives) == 1
            else np.stack(derivatives, axis=-1)
        )
    return groups


def _empty_grid():
    grid_shape = (3, 3, 3)
    return RationalHermiteGrid3D(
        jnp.array([0.0, 1.0, 2.0]),
        jnp.array([0.0, 2.0, 4.0]),
        jnp.array([0.0, 3.0, 6.0]),
        jnp.zeros(grid_shape),
        jnp.zeros(grid_shape + (3,)),
        jnp.zeros(grid_shape + (6,)),
        jnp.zeros(grid_shape + (7,)),
        jnp.zeros(grid_shape + (6,)),
        jnp.zeros(grid_shape + (3,)),
        jnp.zeros(grid_shape),
    )


class RationalHermiteGrid3DTest(unittest.TestCase):
    def test_validates_coordinates_and_group_shapes(self):
        grid = _empty_grid()
        self.assertEqual(grid.shape, ())
        self.assertEqual(grid.grid_shape, (3, 3, 3))
        self.assertEqual(grid.segment_capacity, 4)

        with self.assertRaisesRegex(Exception, "strictly increasing"):
            RationalHermiteGrid3D(
                jnp.array([0.0, 0.0]),
                grid.y,
                grid.z,
                grid.f[:2],
                grid.d1[:2],
                grid.d2[:2],
                grid.d3[:2],
                grid.d4[:2],
                grid.d5[:2],
                grid.d6[:2],
            )
        with self.assertRaisesRegex(ValueError, "d1 must have shape"):
            RationalHermiteGrid3D(
                grid.x,
                grid.y,
                grid.z,
                grid.f,
                jnp.zeros(grid.grid_shape + (2,)),
                grid.d2,
                grid.d3,
                grid.d4,
                grid.d5,
                grid.d6,
            )

    def test_cell_index_clips_and_uses_half_open_cells(self):
        grid = _empty_grid()
        cases = (
            ([-1.0, -1.0, -1.0], [0, 0, 0]),
            ([0.5, 1.0, 1.5], [0, 0, 0]),
            ([1.0, 2.0, 3.0], [1, 1, 1]),
            ([2.0, 4.0, 6.0], [1, 1, 1]),
            ([9.0, 9.0, 9.0], [1, 1, 1]),
        )
        for point, expected in cases:
            npt.assert_array_equal(grid.cell_index(jnp.asarray(point)), expected)

        actual = jax.jit(lambda value: grid.cell_index(value))(
            jnp.array([1.0, 1.0, 4.0])
        )
        npt.assert_array_equal(actual, [1, 0, 1])

    def test_recovers_nonuniform_cell_patch_and_evaluates(self):
        rng = np.random.default_rng(4)
        values = rng.normal(scale=0.2, size=(4, 4, 4)).astype(np.float32)
        weights = rng.uniform(0.7, 1.3, size=(4, 4, 4)).astype(np.float32)
        indices = np.indices((4, 4, 4))
        corners = np.ones((4, 4, 4), dtype=bool)
        for axis in range(3):
            corners &= (indices[axis] == 0) | (indices[axis] == 3)
        weights = np.where(corners, 1.0, weights)
        local_groups = _endpoint_groups(values, weights)
        widths = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        physical_groups = [local_groups[0]]
        for total, group in enumerate(local_groups[1:], start=1):
            scales = np.asarray(
                [np.prod(widths ** np.asarray(alpha)) for alpha in _multi_indices(total)]
            )
            physical_groups.append(
                group / (scales[0] if scales.size == 1 else scales)
            )

        grid = RationalHermiteGrid3D(
            jnp.array([1.0, 3.0]),
            jnp.array([-2.0, 1.0]),
            jnp.array([4.0, 8.0]),
            *(
                jnp.asarray(np.stack((group, 2.0 * group)))
                for group in physical_groups
            ),
        )
        patch = grid.cell_interpolant(jnp.array([0, 0, 0]))
        compiled_patch = jax.jit(lambda index: grid.cell_interpolant(index))(
            jnp.array([0, 0, 0])
        )
        expected = RationalBernstein3D(values, weights)
        local = jnp.array([0.2, 0.4, 0.7])
        physical = jnp.array([1.4, -0.8, 6.8])
        expected_value = expected(local[0], local[1], local[2])

        self.assertEqual(grid.shape, (2,))
        self.assertEqual(patch.shape, (2,))
        npt.assert_allclose(patch.weights, np.stack((weights, weights)), atol=8e-4)
        npt.assert_allclose(compiled_patch.weights, patch.weights, atol=2e-4)
        npt.assert_allclose(
            grid(physical), [expected_value, 2.0 * expected_value], atol=2e-4
        )
        npt.assert_allclose(
            jax.jit(lambda point: grid(point))(physical),
            [expected_value, 2.0 * expected_value],
            atol=2e-4,
        )

    def test_split_segment_is_ordered_masked_and_jittable(self):
        grid = _empty_grid()
        result = grid.split_segment(
            jnp.array([-1.0, -1.0, -1.0]), jnp.array([3.0, 5.0, 7.0])
        )
        self.assertIsInstance(result, GridSegment3D)
        self.assertEqual(result.cell_indices.shape, (4, 3))
        self.assertEqual(result.local_endpoints.shape, (4, 2, 3))
        self.assertEqual(result.valid_mask.shape, (4,))
        npt.assert_array_equal(result.valid_mask, [True, True, False, False])
        npt.assert_array_equal(result.cell_indices[:2], [[0, 0, 0], [1, 1, 1]])
        npt.assert_allclose(result.local_endpoints[0], [[0, 0, 0], [1, 1, 1]])
        npt.assert_allclose(result.local_endpoints[1], [[0, 0, 0], [1, 1, 1]])

        compiled = jax.jit(lambda start, end: grid.split_segment(start, end))
        reverse = compiled(jnp.array([2.0, 1.0, 1.5]), jnp.array([0.0, 1.0, 1.5]))
        npt.assert_array_equal(reverse.valid_mask, [True, True, False, False])
        npt.assert_array_equal(reverse.cell_indices[:2], [[1, 0, 0], [0, 0, 0]])
        npt.assert_allclose(reverse.local_endpoints[0, :, 0], [1.0, 0.0])
        npt.assert_allclose(reverse.local_endpoints[1, :, 0], [1.0, 0.0])

    def test_split_segment_handles_grid_planes_and_zero_length(self):
        grid = _empty_grid()
        on_plane = grid.split_segment(
            jnp.array([1.0, 0.5, 0.5]), jnp.array([1.0, 3.5, 0.5])
        )
        npt.assert_array_equal(on_plane.cell_indices[:2], [[1, 0, 0], [1, 1, 0]])
        npt.assert_array_equal(on_plane.valid_mask, [True, True, False, False])

        point = grid.split_segment(jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 2.0, 3.0]))
        npt.assert_array_equal(point.valid_mask, [True, False, False, False])
        npt.assert_array_equal(point.cell_indices[0], [1, 1, 1])
        npt.assert_allclose(point.local_endpoints[0], jnp.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
