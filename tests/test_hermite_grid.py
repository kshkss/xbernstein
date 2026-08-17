import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import HermiteGrid1D, HermiteGrid2D, HermiteGrid3D, HermiteGrid4D


class HermiteGridTest(unittest.TestCase):
    def test_all_dimensions_interpolate_affine_field(self):
        grids = (
            HermiteGrid1D(
                jnp.array([0.0, 2.0, 5.0]),
                jnp.array([0.0, 2.0, 5.0]),
                jnp.ones(3),
            ),
            HermiteGrid2D(
                jnp.array([0.0, 2.0]), jnp.array([-1.0, 3.0]),
                jnp.array([[-1.0, 3.0], [1.0, 5.0]]),
                jnp.ones((2, 2, 2)), jnp.zeros((2, 2)),
            ),
            HermiteGrid3D(
                jnp.array([0.0, 2.0]), jnp.array([-1.0, 3.0]), jnp.array([2.0, 5.0]),
                sum(jnp.meshgrid(jnp.array([0.0, 2.0]), jnp.array([-1.0, 3.0]), jnp.array([2.0, 5.0]), indexing="ij")),
                jnp.ones((2, 2, 2, 3)),
                jnp.zeros((2, 2, 2, 3)), jnp.zeros((2, 2, 2)),
            ),
            HermiteGrid4D(
                jnp.array([0.0, 2.0]), jnp.array([-1.0, 3.0]),
                jnp.array([2.0, 5.0]), jnp.array([-2.0, 4.0]),
                sum(jnp.meshgrid(jnp.array([0.0, 2.0]), jnp.array([-1.0, 3.0]),
                                 jnp.array([2.0, 5.0]), jnp.array([-2.0, 4.0]), indexing="ij")),
                jnp.ones((2, 2, 2, 2, 4)),
                jnp.zeros((2, 2, 2, 2, 6)), jnp.zeros((2, 2, 2, 2, 4)),
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
                npt.assert_allclose(jax.jit(lambda p: grid(p))(point), expected, atol=1e-5)

    def test_validates_shapes_and_clips(self):
        grid = HermiteGrid3D(
            jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 2.0]), jnp.array([0.0, 3.0]),
            jnp.zeros((3, 2, 2)), jnp.zeros((3, 2, 2, 3)),
            jnp.zeros((3, 2, 2, 3)), jnp.zeros((3, 2, 2)),
        )
        npt.assert_array_equal(grid.cell_index(jnp.array([-1.0, 4.0, 9.0])), [0, 0, 0])
        npt.assert_array_equal(grid.cell_index(jnp.array([2.0, 2.0, 3.0])), [1, 0, 0])
        with self.assertRaisesRegex(ValueError, "d1 must have shape"):
            HermiteGrid3D(
                grid.axes[0], grid.axes[1], grid.axes[2], grid.f,
                jnp.zeros((3, 2, 2, 2)), grid.groups[2], grid.groups[3],
            )

    def test_split_segment_is_available_only_from_2d(self):
        grid = HermiteGrid2D(
            jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 1.0, 2.0]),
            jnp.zeros((3, 3)), jnp.zeros((3, 3, 2)), jnp.zeros((3, 3)),
        )
        result = grid.split_segment(jnp.array([-1.0, -1.0]), jnp.array([3.0, 3.0]))
        npt.assert_array_equal(result.valid_mask, [True, True, False])
        self.assertFalse(hasattr(HermiteGrid1D(jnp.array([0.0, 1.0]), jnp.zeros(2), jnp.zeros(2)), "split_segment"))


if __name__ == "__main__":
    unittest.main()
