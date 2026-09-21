import unittest

import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import (
    C0Grid1D,
    P1C0Grid1D,
    P1C0Grid2D,
    P1C0Grid3D,
    P1C0Grid4D,
    P2C0Grid1D,
    P2C0Grid2D,
    P2C0Grid3D,
    P2C0Grid4D,
    P3C0Grid2D,
    P3C0Grid3D,
    P3C0Grid4D,
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
        npt.assert_array_equal(left(jnp.ones_like(v), v), right(jnp.zeros_like(v), v))

    def test_p3_grid_shares_the_entire_edge_between_two_cells(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 3.0])
        f = jnp.arange(7 * 4, dtype=jnp.float32).reshape(7, 4)
        grid = P3C0Grid2D(x, y, f)
        self.assertEqual(grid.dof_shape, (7, 4))

        left = grid.cell_interpolant(jnp.array([0, 0]))
        right = grid.cell_interpolant(jnp.array([1, 0]))
        v = jnp.linspace(0.0, 1.0, 7)
        npt.assert_array_equal(left(jnp.ones_like(v), v), right(jnp.zeros_like(v), v))

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


class C0GridSegmentGridTest(unittest.TestCase):
    def test_reproduces_an_affine_field_exactly(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0, 3.0])
        grid = P1C0Grid2D(x, y, sum(jnp.meshgrid(x, y, indexing="ij")))
        start = jnp.array([0.25, 0.5])
        end = jnp.array([1.75, 2.5])

        result = grid.segment_grid(start, end)
        self.assertIsInstance(result, C0Grid1D)

        delta = end - start
        length = float(jnp.linalg.norm(delta))
        for t in jnp.linspace(0.0, length, 9):
            point = start + (t / length) * delta
            npt.assert_allclose(result(jnp.array([t])), jnp.sum(point), atol=1e-5)

    def test_matches_direct_evaluation_across_a_grid_line_crossing(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 3.0])
        f = jnp.arange(5 * 3, dtype=jnp.float32).reshape(5, 3)
        grid = P2C0Grid2D(x, y, f)
        start = jnp.array([0.25, 0.5])
        end = jnp.array([1.75, 2.5])

        result = grid.segment_grid(start, end)
        self.assertEqual(result.degree, grid.dimension * grid.degree)

        delta = end - start
        length = float(jnp.linalg.norm(delta))
        for t in jnp.linspace(0.0, length, 13):
            point = start + (t / length) * delta
            npt.assert_allclose(
                result(jnp.array([t])), grid(point), atol=1e-4, rtol=1e-4
            )

    def test_segment_confined_to_a_single_cell_has_one_piece(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0])
        z = jnp.array([0.0, 1.0])
        f = jnp.arange(5 * 3 * 3, dtype=jnp.float32).reshape(5, 3, 3)
        grid = P2C0Grid3D(x, y, z, f)
        start = jnp.array([0.1, 0.1, 0.1])
        end = jnp.array([0.9, 0.9, 0.9])

        result = grid.segment_grid(start, end)
        self.assertEqual(result.dof_shape, (grid.dimension * grid.degree + 1,))

    def test_not_available_for_1d(self):
        self.assertFalse(
            hasattr(P1C0Grid1D(jnp.array([0.0, 1.0]), jnp.zeros(2)), "segment_grid")
        )

    def test_degenerate_segment_raises(self):
        grid = P1C0Grid2D(
            jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 1.0, 2.0]), jnp.zeros((3, 3))
        )
        with self.assertRaises(ValueError):
            grid.segment_grid(jnp.array([1.0, 1.0]), jnp.array([1.0, 1.0]))


class C0GridIntegrateOutTest(unittest.TestCase):
    def test_matches_closed_form_bilinear_integral_2d_to_1d(self):
        a, b, c, d = 1.0, 2.0, -3.0, 0.5
        x = jnp.array([0.0, 1.0, 2.5])
        y = jnp.array([-1.0, 0.0, 4.0])
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        f = a + b * X + c * Y + d * X * Y
        grid = P1C0Grid2D(x, y, f)

        result = grid.integrate_out(axis=0)
        self.assertIsInstance(result, P1C0Grid1D)

        x0, xn = float(x[0]), float(x[-1])
        coefficient_a = (xn - x0) * (a + b * (x0 + xn) / 2.0)
        coefficient_b = (xn - x0) * (c + d * (x0 + xn) / 2.0)
        for yj in jnp.linspace(y[0], y[-1], 9):
            npt.assert_allclose(
                result(jnp.array([yj])),
                coefficient_a + coefficient_b * yj,
                atol=1e-4,
            )

    def test_nonuniform_cell_widths_sum_correctly(self):
        x = jnp.array([0.0, 1.0, 1.5, 4.0])
        y = jnp.array([0.0, 2.0])
        f = jnp.arange(7 * 3, dtype=jnp.float32).reshape(7, 3)
        grid = P2C0Grid2D(x, y, f)

        expected = jnp.zeros(3)
        for c in range(x.shape[0] - 1):
            width = x[c + 1] - x[c]
            polynomial = grid.cell_interpolant(jnp.array([c, 0])).integrate_out(axis=0)
            expected = expected + width * polynomial.c

        result = grid.integrate_out(axis=0)
        self.assertIsInstance(result, P2C0Grid1D)
        npt.assert_allclose(result.f, expected, atol=1e-4)

    def test_preserves_degree_and_drops_one_dimension_3d_to_2d(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0])
        z = jnp.array([0.0, 1.0, 3.0])
        f = jnp.zeros((5, 3, 5))
        grid = P2C0Grid3D(x, y, z, f)

        result = grid.integrate_out(axis=1)
        self.assertIsInstance(result, P2C0Grid2D)
        self.assertEqual(result.dimension, 2)
        self.assertEqual(result.degree, 2)
        npt.assert_array_equal(result.axes[0], x)
        npt.assert_array_equal(result.axes[1], z)

    def test_preserves_degree_and_drops_one_dimension_4d_to_3d(self):
        x = jnp.array([0.0, 1.0])
        y = jnp.array([0.0, 1.0])
        z = jnp.array([0.0, 1.0])
        w = jnp.array([0.0, 1.0])
        f = jnp.zeros((4, 4, 4, 4))
        grid = P3C0Grid4D(x, y, z, w, f)

        result = grid.integrate_out(axis=3)
        self.assertIsInstance(result, P3C0Grid3D)
        self.assertEqual(result.dimension, 3)
        self.assertEqual(result.degree, 3)

    def test_invalid_axis_raises(self):
        grid = P1C0Grid2D(
            jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 1.0]), jnp.zeros((3, 2))
        )
        with self.assertRaises(ValueError):
            grid.integrate_out(axis=-1)
        with self.assertRaises(ValueError):
            grid.integrate_out(axis=2)

    def test_not_available_for_1d(self):
        self.assertFalse(
            hasattr(P1C0Grid1D(jnp.array([0.0, 1.0]), jnp.zeros(2)), "integrate_out")
        )
        self.assertFalse(
            hasattr(
                C0Grid1D(jnp.array([0.0, 1.0]), jnp.zeros(3), degree=2),
                "integrate_out",
            )
        )


if __name__ == "__main__":
    unittest.main()
