import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import (
    HermiteGrid1D,
    HermiteGrid2D,
    HermiteGrid3D,
    HermiteGrid4D,
    hermite_interpolate_1d,
    hermite_interpolate_2d,
    hermite_interpolate_3d,
    hermite_interpolate_4d,
)


def _multi_indices(dimension, total):
    return sorted(
        (
            alpha
            for alpha in itertools.product(range(2), repeat=dimension)
            if sum(alpha) == total
        ),
        reverse=True,
    )


def _quadratic_groups(axes):
    coordinates = jnp.meshgrid(*axes, indexing="ij")
    dimension = len(axes)
    groups = []
    for total in range(dimension + 1):
        derivatives = []
        for alpha in _multi_indices(dimension, total):
            value = jnp.ones(tuple(axis.size for axis in axes))
            for coordinate, order in zip(coordinates, alpha):
                value = value * (
                    1.0 + coordinate + coordinate**2
                    if order == 0
                    else 1.0 + 2.0 * coordinate
                )
            derivatives.append(value)
        groups.append(
            derivatives[0] if len(derivatives) == 1 else jnp.stack(derivatives, axis=-1)
        )
    return tuple(groups)


def _cell_groups(axes, groups, cell_index):
    dimension = len(axes)
    widths = jnp.asarray(
        [axis[index + 1] - axis[index] for axis, index in zip(axes, cell_index)]
    )
    local = []
    for total, group in enumerate(groups):
        vertices = group[tuple(slice(index, index + 2) for index in cell_index)]
        alphas = _multi_indices(dimension, total)
        scales = jnp.asarray(
            [jnp.prod(widths ** jnp.asarray(alpha)) for alpha in alphas]
        )
        local.append(
            vertices * scales[0]
            if len(alphas) == 1
            else vertices * scales.reshape((1,) * dimension + (len(alphas),))
        )
    return tuple(local)


class HermiteGridTest(unittest.TestCase):
    def test_stores_cell_coefficients_and_reconstructs_all_vertex_groups(self):
        classes = (HermiteGrid1D, HermiteGrid2D, HermiteGrid3D, HermiteGrid4D)
        interpolators = (
            hermite_interpolate_1d,
            hermite_interpolate_2d,
            hermite_interpolate_3d,
            hermite_interpolate_4d,
        )
        all_axes = (
            jnp.array([0.0, 1.0, 3.0]),
            jnp.array([-1.0, 2.0]),
            jnp.array([0.0, 2.0]),
            jnp.array([-2.0, 1.0]),
        )

        for dimension, (grid_type, interpolate) in enumerate(
            zip(classes, interpolators), start=1
        ):
            with self.subTest(dimension=dimension):
                axes = all_axes[:dimension]
                groups = _quadratic_groups(axes)
                grid = grid_type(*axes, *groups)
                cell_shape = tuple(axis.size - 1 for axis in axes)
                self.assertEqual(
                    grid.coefficients.shape,
                    cell_shape + (4,) * dimension,
                )
                self.assertFalse(hasattr(grid, "groups"))

                cell_index = (1,) + (0,) * (dimension - 1)
                expected = interpolate(*_cell_groups(axes, groups, cell_index))
                npt.assert_array_equal(
                    grid.cell_interpolant(jnp.asarray(cell_index)).c,
                    expected.c,
                )

                for total, source in enumerate(groups):
                    reconstructed = grid.f if total == 0 else getattr(grid, f"d{total}")
                    npt.assert_allclose(
                        reconstructed,
                        source,
                        rtol=2e-5,
                        atol=3e-4,
                    )

    def test_all_dimensions_interpolate_affine_field(self):
        grids = (
            HermiteGrid1D(
                jnp.array([0.0, 2.0, 5.0]),
                jnp.array([0.0, 2.0, 5.0]),
                jnp.ones(3),
            ),
            HermiteGrid2D(
                jnp.array([0.0, 2.0]),
                jnp.array([-1.0, 3.0]),
                jnp.array([[-1.0, 3.0], [1.0, 5.0]]),
                jnp.ones((2, 2, 2)),
                jnp.zeros((2, 2)),
            ),
            HermiteGrid3D(
                jnp.array([0.0, 2.0]),
                jnp.array([-1.0, 3.0]),
                jnp.array([2.0, 5.0]),
                sum(
                    jnp.meshgrid(
                        jnp.array([0.0, 2.0]),
                        jnp.array([-1.0, 3.0]),
                        jnp.array([2.0, 5.0]),
                        indexing="ij",
                    )
                ),
                jnp.ones((2, 2, 2, 3)),
                jnp.zeros((2, 2, 2, 3)),
                jnp.zeros((2, 2, 2)),
            ),
            HermiteGrid4D(
                jnp.array([0.0, 2.0]),
                jnp.array([-1.0, 3.0]),
                jnp.array([2.0, 5.0]),
                jnp.array([-2.0, 4.0]),
                sum(
                    jnp.meshgrid(
                        jnp.array([0.0, 2.0]),
                        jnp.array([-1.0, 3.0]),
                        jnp.array([2.0, 5.0]),
                        jnp.array([-2.0, 4.0]),
                        indexing="ij",
                    )
                ),
                jnp.ones((2, 2, 2, 2, 4)),
                jnp.zeros((2, 2, 2, 2, 6)),
                jnp.zeros((2, 2, 2, 2, 4)),
                jnp.zeros((2, 2, 2, 2)),
            ),
        )
        points = (
            jnp.array([1.25]),
            jnp.array([1.25, 0.75]),
            jnp.array([1.25, 0.75, 3.25]),
            jnp.array([1.25, 0.75, 3.25, 1.0]),
        )
        for grid, point in zip(grids, points):
            with self.subTest(dimension=grid.dimension):
                expected = jnp.sum(point)
                npt.assert_allclose(grid(point), expected, atol=1e-5)
                npt.assert_allclose(
                    jax.jit(lambda p: grid(p))(point), expected, atol=1e-5
                )

    def test_validates_shapes_and_clips(self):
        d2 = jnp.zeros((3, 2, 2, 3))
        d3 = jnp.zeros((3, 2, 2))
        grid = HermiteGrid3D(
            jnp.array([0.0, 1.0, 2.0]),
            jnp.array([0.0, 2.0]),
            jnp.array([0.0, 3.0]),
            jnp.zeros((3, 2, 2)),
            jnp.zeros((3, 2, 2, 3)),
            d2,
            d3,
        )
        npt.assert_array_equal(grid.cell_index(jnp.array([-1.0, 4.0, 9.0])), [0, 0, 0])
        npt.assert_array_equal(grid.cell_index(jnp.array([2.0, 2.0, 3.0])), [1, 0, 0])
        with self.assertRaisesRegex(ValueError, "d1 must have shape"):
            HermiteGrid3D(
                grid.axes[0],
                grid.axes[1],
                grid.axes[2],
                grid.f,
                jnp.zeros((3, 2, 2, 2)),
                d2,
                d3,
            )

    def test_split_segment_is_available_only_from_2d(self):
        grid = HermiteGrid2D(
            jnp.array([0.0, 1.0, 2.0]),
            jnp.array([0.0, 1.0, 2.0]),
            jnp.zeros((3, 3)),
            jnp.zeros((3, 3, 2)),
            jnp.zeros((3, 3)),
        )
        result = grid.split_segment(jnp.array([-1.0, -1.0]), jnp.array([3.0, 3.0]))
        npt.assert_array_equal(result.valid_mask, [True, True, False])
        self.assertFalse(
            hasattr(
                HermiteGrid1D(jnp.array([0.0, 1.0]), jnp.zeros(2), jnp.zeros(2)),
                "split_segment",
            )
        )


if __name__ == "__main__":
    unittest.main()
