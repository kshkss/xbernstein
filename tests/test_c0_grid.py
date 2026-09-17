import unittest

import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import (
    P1C0Grid1D,
    P1C0Grid2D,
    P1C0Grid3D,
    P1C0Grid4D,
    P2C0Grid2D,
    P2C0Grid3D,
    P2C0Grid4D,
    P3C0Grid2D,
)


class C0GridAffineReproductionTest(unittest.TestCase):
    def test_p1_grids_reproduce_an_affine_field_in_every_dimension(self):
        axes = (
            jnp.array([0.0, 2.0, 5.0]),
            jnp.array([-1.0, 3.0]),
            jnp.array([2.0, 5.0]),
            jnp.array([-2.0, 4.0]),
        )
        grids = (
            P1C0Grid1D(axes[0], axes[0]),
            P1C0Grid2D(
                axes[0],
                axes[1],
                sum(jnp.meshgrid(axes[0], axes[1], indexing="ij")),
            ),
            P1C0Grid3D(
                axes[0],
                axes[1],
                axes[2],
                sum(jnp.meshgrid(axes[0], axes[1], axes[2], indexing="ij")),
            ),
            P1C0Grid4D(
                axes[0],
                axes[1],
                axes[2],
                axes[3],
                sum(jnp.meshgrid(*axes, indexing="ij")),
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
                self.assertEqual(grid.dof_shape, grid.grid_shape)
                npt.assert_allclose(grid(point), jnp.sum(point), atol=1e-5)


class C0GridFaceContinuityTest(unittest.TestCase):
    def test_p2_grid_shares_the_entire_edge_between_two_cells(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 3.0])
        f = jnp.arange(5 * 3, dtype=jnp.float32).reshape(5, 3)
        grid = P2C0Grid2D(x, y, f)
        self.assertEqual(grid.dof_shape, (5, 3))

        left = grid.cell_interpolant(jnp.array([0, 0]))
        right = grid.cell_interpolant(jnp.array([1, 0]))
        v = jnp.linspace(0.0, 1.0, 7)
        npt.assert_array_equal(
            left(jnp.ones_like(v), v), right(jnp.zeros_like(v), v)
        )

    def test_p3_grid_shares_the_entire_edge_between_two_cells(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 3.0])
        f = jnp.arange(7 * 4, dtype=jnp.float32).reshape(7, 4)
        grid = P3C0Grid2D(x, y, f)
        self.assertEqual(grid.dof_shape, (7, 4))

        left = grid.cell_interpolant(jnp.array([0, 0]))
        right = grid.cell_interpolant(jnp.array([1, 0]))
        v = jnp.linspace(0.0, 1.0, 7)
        npt.assert_array_equal(
            left(jnp.ones_like(v), v), right(jnp.zeros_like(v), v)
        )

    def test_vertex_entries_equal_the_true_function_value(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 3.0])
        f = jnp.arange(5 * 3, dtype=jnp.float32).reshape(5, 3)
        grid = P2C0Grid2D(x, y, f)
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                npt.assert_allclose(
                    grid(jnp.array([xi, yj])), f[2 * i, 2 * j], atol=1e-5
                )


class C0GridValidationAndSmokeTest(unittest.TestCase):
    def test_rejects_wrong_f_shape(self):
        with self.assertRaisesRegex(ValueError, "f must end with shape"):
            P2C0Grid2D(
                jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 3.0]), jnp.zeros((5, 2))
            )

    def test_split_segment_is_available_only_from_2d(self):
        grid = P1C0Grid2D(
            jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 1.0, 2.0]), jnp.zeros((3, 3))
        )
        result = grid.split_segment(jnp.array([-1.0, -1.0]), jnp.array([3.0, 3.0]))
        npt.assert_array_equal(result.valid_mask, [True, True, False])
        self.assertFalse(
            hasattr(P1C0Grid1D(jnp.array([0.0, 1.0]), jnp.zeros(2)), "split_segment")
        )

    def test_3d_and_4d_construct_and_evaluate(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0])
        z = jnp.array([0.0, 1.0])
        w = jnp.array([0.0, 1.0])

        grid3d = P2C0Grid3D(x, y, z, jnp.zeros((5, 3, 3)))
        self.assertEqual(grid3d.dof_shape, (5, 3, 3))
        npt.assert_allclose(grid3d(jnp.array([0.5, 0.5, 0.5])), 0.0)

        grid4d = P2C0Grid4D(x, y, z, w, jnp.zeros((5, 3, 3, 3)))
        self.assertEqual(grid4d.dof_shape, (5, 3, 3, 3))
        npt.assert_allclose(grid4d(jnp.array([0.5, 0.5, 0.5, 0.5])), 0.0)


if __name__ == "__main__":
    unittest.main()
