import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import (
    QuinticHermiteGrid1D,
    QuinticHermiteGrid2D,
    QuinticHermiteGrid3D,
    QuinticHermiteGrid4D,
)


def _multi_indices(dimension, total):
    return sorted(
        (
            alpha
            for alpha in itertools.product(range(3), repeat=dimension)
            if sum(alpha) == total
        ),
        reverse=True,
    )


def _quadratic_groups(axes):
    coordinates = jnp.meshgrid(*axes, indexing="ij")
    dimension = len(axes)
    groups = []
    for total in range(2 * dimension + 1):
        derivatives = []
        for alpha in _multi_indices(dimension, total):
            value = jnp.ones(tuple(axis.size for axis in axes))
            for coordinate, order in zip(coordinates, alpha):
                if order == 0:
                    value = value * (1.0 + coordinate + coordinate**2)
                elif order == 1:
                    value = value * (1.0 + 2.0 * coordinate)
                else:
                    value = value * 2.0
            derivatives.append(value)
        groups.append(
            derivatives[0] if len(derivatives) == 1 else jnp.stack(derivatives, axis=-1)
        )
    return tuple(groups)


class QuinticHermiteGridTest(unittest.TestCase):
    def test_all_dimensions_reproduce_quadratic_tensor_field(self):
        classes = (
            QuinticHermiteGrid1D,
            QuinticHermiteGrid2D,
            QuinticHermiteGrid3D,
            QuinticHermiteGrid4D,
        )
        all_axes = (
            jnp.array([0.0, 1.0, 3.0]),
            jnp.array([-1.0, 2.0]),
            jnp.array([0.0, 2.0]),
            jnp.array([-2.0, 1.0]),
        )
        point = jnp.array([2.0, 0.25, 0.75, -0.5])

        for dimension, grid_type in enumerate(classes, start=1):
            with self.subTest(dimension=dimension):
                axes = all_axes[:dimension]
                groups = _quadratic_groups(axes)
                grid = grid_type(*axes, *groups)
                sample = point[:dimension]
                expected = jnp.prod(1.0 + sample + sample**2)

                npt.assert_allclose(grid(sample), expected, atol=2e-4)
                if dimension == 1:
                    npt.assert_allclose(
                        jax.jit(lambda value: grid(value))(sample),
                        expected,
                        atol=2e-4,
                    )
                npt.assert_array_equal(
                    jnp.atleast_1d(
                        grid.cell_interpolant(grid.cell_index(sample)).order
                    ),
                    jnp.full(dimension, 5),
                )
                npt.assert_allclose(getattr(grid, f"d{2 * dimension}"), groups[-1])
                self.assertEqual(grid.jet_size, 3)

    def test_nonuniform_1d_cells_are_c2_in_physical_coordinates(self):
        axis = jnp.array([0.0, 1.0, 3.0])
        grid = QuinticHermiteGrid1D(axis, *_quadratic_groups((axis,)))
        left = grid.cell_interpolant(jnp.array([0]))
        right = grid.cell_interpolant(jnp.array([1]))
        left_width = axis[1] - axis[0]
        right_width = axis[2] - axis[1]

        npt.assert_allclose(left(1.0), right(0.0), atol=1e-6)
        npt.assert_allclose(
            left.deriv()(1.0) / left_width,
            right.deriv()(0.0) / right_width,
            atol=2e-5,
        )
        npt.assert_allclose(
            left.deriv(m=2)(1.0) / left_width**2,
            right.deriv(m=2)(0.0) / right_width**2,
            atol=2e-5,
        )

    def test_batch_shapes_validation_clipping_and_derivative_properties(self):
        axis = jnp.array([0.0, 1.0, 3.0])
        groups = _quadratic_groups((axis,))
        batched = tuple(jnp.stack((group, 2.0 * group)) for group in groups)
        grid = QuinticHermiteGrid1D(axis, *batched)
        expected = 1.0 + 3.0 + 3.0**2

        self.assertEqual(grid.shape, (2,))
        npt.assert_allclose(grid(jnp.array([9.0])), [expected, 2.0 * expected])
        npt.assert_allclose(grid.d2, batched[2])

        axes = (axis, jnp.array([0.0, 2.0]), jnp.array([-1.0, 1.0]))
        groups_3d = _quadratic_groups(axes)
        grid_3d = QuinticHermiteGrid3D(*axes, *groups_3d)
        npt.assert_allclose(grid_3d.d5, groups_3d[5])
        npt.assert_allclose(grid_3d.d6, groups_3d[6])
        with self.assertRaisesRegex(ValueError, "d2 must have shape"):
            QuinticHermiteGrid3D(
                *axes,
                groups_3d[0],
                groups_3d[1],
                jnp.zeros(axes[0].shape + axes[1].shape + axes[2].shape + (5,)),
                *groups_3d[3:],
            )

    def test_split_segment_is_available_only_from_2d(self):
        axes = (jnp.array([0.0, 1.0, 2.0]),) * 2
        grid = QuinticHermiteGrid2D(*axes, *_quadratic_groups(axes))
        result = grid.split_segment(jnp.array([-1.0, -1.0]), jnp.array([3.0, 3.0]))

        npt.assert_array_equal(result.valid_mask, [True, True, False])
        one_axis = axes[:1]
        self.assertFalse(
            hasattr(
                QuinticHermiteGrid1D(*one_axis, *_quadratic_groups(one_axis)),
                "split_segment",
            )
        )
        self.assertTrue(hasattr(QuinticHermiteGrid3D, "split_segment"))
        self.assertTrue(hasattr(QuinticHermiteGrid4D, "split_segment"))


if __name__ == "__main__":
    unittest.main()
