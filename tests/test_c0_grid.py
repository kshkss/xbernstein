import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import (
    Bernstein2D,
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
    maximize,
    minimize,
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
        # Now caught earlier, at the constructor's own precise shape
        # expression (Float[..., "*batch 2*nx-1 2*ny-1"]), before it would
        # even reach _initialize's own "f must end with shape" check.
        from jaxtyping import TypeCheckError

        with self.assertRaises(TypeCheckError):
            P2C0Grid2D(
                jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 3.0]), jnp.zeros((5, 2))
            )

    def test_rejects_non_positive_degree(self):
        # degree=0 happens to produce a dof-shape expression that matches
        # this f's actual shape (0*(nx-1)+1 == 1), so it slips past the
        # constructor's own type check and is instead caught by
        # _initialize's explicit value check.
        with self.assertRaisesRegex(ValueError, "degree must be"):
            C0Grid1D(jnp.array([0.0, 1.0]), jnp.zeros(1), degree=0)
        # A different non-positive degree whose dof-shape expression does
        # NOT match this f's shape is instead caught earlier, at the
        # constructor's own type boundary.
        from jaxtyping import TypeCheckError

        with self.assertRaises(TypeCheckError):
            C0Grid1D(jnp.array([0.0, 1.0]), jnp.zeros(1), degree=-1)

    def test_rejects_non_integer_degree(self):
        from jaxtyping import TypeCheckError

        with self.assertRaises(TypeCheckError):
            C0Grid1D(jnp.array([0.0, 1.0]), jnp.zeros(1), degree=1.5)
        with self.assertRaises(TypeCheckError):
            C0Grid1D(jnp.array([0.0, 1.0]), jnp.zeros(1), degree=2.0)

    def test_call_and_cell_index_reject_a_non_array_point(self):
        from jaxtyping import TypeCheckError

        grid = P1C0Grid1D(jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 1.0, 2.0]))
        with self.assertRaises(TypeCheckError):
            grid([0.5])
        with self.assertRaises(TypeCheckError):
            grid.cell_index([0.5])

    def test_split_segment_and_segment_grid_reject_mismatched_start_end(self):
        from jaxtyping import TypeCheckError

        grid = P1C0Grid2D(
            jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 1.0, 2.0]), jnp.zeros((3, 3))
        )
        with self.assertRaises(TypeCheckError):
            grid.split_segment(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0, 1.0]))
        with self.assertRaises(TypeCheckError):
            grid.segment_grid(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0, 1.0]))

    def test_minimize_and_integrate_out_reject_non_integer_scalars(self):
        from jaxtyping import TypeCheckError

        grid = P1C0Grid2D(
            jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 1.0, 2.0]), jnp.zeros((3, 3))
        )
        with self.assertRaises(TypeCheckError):
            grid.minimize(max_steps=1.5)
        with self.assertRaises(TypeCheckError):
            grid.integrate_out(axis=0.5)

    def test_constructors_reject_f_with_the_wrong_degree_dependent_shape(self):
        # Representative check that the precise per-axis shape expression
        # (e.g. "2*nx-1") actually fires, not just "is an array": degree-1
        # sized f passed to a degree-2 constructor.
        from jaxtyping import TypeCheckError

        x = jnp.array([0.0, 1.0, 2.0])
        with self.assertRaises(TypeCheckError):
            P2C0Grid1D(x, jnp.zeros(3))  # degree=2 needs 2*3-1=5, not 3
        y = jnp.array([0.0, 1.0])
        with self.assertRaises(TypeCheckError):
            P3C0Grid2D(x, y, jnp.zeros((5, 3)))  # degree=3 needs (7, 4)

    def test_constructors_reject_non_float_arrays(self):
        from jaxtyping import TypeCheckError

        x_int = jnp.array([0, 1, 2])
        with self.assertRaises(TypeCheckError):
            P1C0Grid1D(x_int, jnp.array([0.0, 1.0, 2.0]))
        with self.assertRaises(TypeCheckError):
            P1C0Grid1D(jnp.array([0.0, 1.0, 2.0]), jnp.array([0, 1, 2]))

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

    def test_result_is_always_segment_capacity_sized(self):
        # segment_grid must be jit-traceable, so its output shape cannot
        # depend on how many cells the segment actually crosses — it is
        # always padded out to the worst-case `segment_capacity`, even for
        # a segment confined to a single cell.
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0])
        z = jnp.array([0.0, 1.0])
        f = jnp.arange(5 * 3 * 3, dtype=jnp.float32).reshape(5, 3, 3)
        grid = P2C0Grid3D(x, y, z, f)
        start = jnp.array([0.1, 0.1, 0.1])
        end = jnp.array([0.9, 0.9, 0.9])

        result = grid.segment_grid(start, end)
        expected_size = grid.dimension * grid.degree * grid.segment_capacity + 1
        self.assertEqual(result.dof_shape, (expected_size,))

        delta = end - start
        length = float(jnp.linalg.norm(delta))
        for t in jnp.linspace(0.0, length, 5):
            point = start + (t / length) * delta
            npt.assert_allclose(result(jnp.array([t])), grid(point), atol=1e-4)

    def test_matches_the_true_endpoint_value_exactly_at_the_real_length(self):
        # Regression test: padding slots default to the grid's (0,...,0)
        # corner value, unrelated to the segment's real endpoint. Querying
        # exactly at t = length must not fall through to that corner value
        # — the padded tail must be a flat, continuous extension of the
        # real endpoint instead.
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 3.0])
        f = jnp.arange(5 * 3, dtype=jnp.float32).reshape(5, 3)
        grid = P2C0Grid2D(x, y, f)
        start = jnp.array([0.25, 0.5])
        end = jnp.array([1.75, 2.5])

        result = grid.segment_grid(start, end)
        length = float(jnp.linalg.norm(end - start))

        npt.assert_allclose(result(jnp.array([length])), grid(end), atol=1e-4)
        # And well past the real length, it should stay flat at that value.
        npt.assert_allclose(result(jnp.array([length + 10.0])), grid(end), atol=1e-4)

    def test_is_jit_traceable(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 3.0])
        f = jnp.arange(5 * 3, dtype=jnp.float32).reshape(5, 3)
        grid = P2C0Grid2D(x, y, f)
        start = jnp.array([0.25, 0.5])
        end = jnp.array([1.75, 2.5])

        eager = grid.segment_grid(start, end)

        @jax.jit
        def run(s, e):
            return grid.segment_grid(s, e)

        jitted = run(start, end)

        npt.assert_allclose(jitted.x, eager.x, atol=1e-6)
        npt.assert_allclose(jitted.f, eager.f, atol=1e-6)

    def test_supports_a_leading_batch_dimension(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0, 3.0])
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        f0 = X + Y
        f1 = 2.0 * X - Y
        grid = P1C0Grid2D(x, y, jnp.stack([f0, f1]))
        start = jnp.array([0.25, 0.5])
        end = jnp.array([1.75, 2.5])

        result = grid.segment_grid(start, end)
        self.assertEqual(result.shape, (2,))

        expected0 = P1C0Grid2D(x, y, f0).segment_grid(start, end)
        expected1 = P1C0Grid2D(x, y, f1).segment_grid(start, end)

        npt.assert_allclose(result.x, expected0.x, atol=1e-6)
        npt.assert_allclose(result.x, expected1.x, atol=1e-6)
        npt.assert_allclose(result.f[0], expected0.f, atol=1e-5)
        npt.assert_allclose(result.f[1], expected1.f, atol=1e-5)

    def test_supports_a_single_cell_batched_grid(self):
        # segment_capacity == 1 here, so the welding scan runs over a
        # zero-length remainder (pieces[1:]/mask[1:] are empty) -- a
        # degenerate case the general batch test above doesn't exercise.
        x = jnp.array([0.0, 1.0])
        y = jnp.array([0.0, 1.0])
        f0 = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        f1 = jnp.array([[1.0, 0.0], [3.0, 2.0]])
        grid = P1C0Grid2D(x, y, jnp.stack([f0, f1]))
        self.assertEqual(grid.segment_capacity, 1)
        start = jnp.array([0.1, 0.1])
        end = jnp.array([0.9, 0.9])

        result = grid.segment_grid(start, end)

        expected0 = P1C0Grid2D(x, y, f0).segment_grid(start, end)
        expected1 = P1C0Grid2D(x, y, f1).segment_grid(start, end)
        npt.assert_allclose(result.f[0], expected0.f, atol=1e-5)
        npt.assert_allclose(result.f[1], expected1.f, atol=1e-5)

    def test_supports_a_batched_grid_with_no_crossings(self):
        # The segment stays inside a single cell of a multi-cell grid, so
        # _split_segment's no_piece fallback path is exercised together
        # with batching.
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0, 2.0])
        f0 = jnp.arange(9.0).reshape(3, 3)
        f1 = jnp.arange(9.0).reshape(3, 3) * 2.0
        grid = P1C0Grid2D(x, y, jnp.stack([f0, f1]))
        start = jnp.array([0.1, 0.1])
        end = jnp.array([0.15, 0.12])

        result = grid.segment_grid(start, end)

        expected0 = P1C0Grid2D(x, y, f0).segment_grid(start, end)
        expected1 = P1C0Grid2D(x, y, f1).segment_grid(start, end)
        npt.assert_allclose(result.f[0], expected0.f, atol=1e-5)
        npt.assert_allclose(result.f[1], expected1.f, atol=1e-5)

    def test_supports_a_multi_axis_batch_shape(self):
        # A rank-2 batch shape, not just a single leading batch axis, to
        # confirm the welding reshape generalizes beyond one batch axis.
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0, 3.0])
        f = jnp.arange(2 * 3 * 3 * 3, dtype=jnp.float32).reshape(2, 3, 3, 3)
        grid = P1C0Grid2D(x, y, f)
        start = jnp.array([0.25, 0.5])
        end = jnp.array([1.75, 2.5])

        result = grid.segment_grid(start, end)
        self.assertEqual(result.shape, (2, 3))

        for i in range(2):
            for j in range(3):
                expected = P1C0Grid2D(x, y, f[i, j]).segment_grid(start, end)
                npt.assert_allclose(result.f[i, j], expected.f, atol=1e-5)

    def test_preserves_narrow_intervals_near_a_grid_line(self):
        # Regression test: a fixed absolute tolerance on interval width
        # would drop this first, genuinely narrow-but-real interval
        # ([0, 1e-7]) as spuriously "too small", shifting the restriction's
        # apparent start to the interior crossing at x=1e-7 instead of the
        # segment's true start at x=0.
        x = jnp.array([0.0, 1e-7, 1.0])
        y = jnp.array([0.0, 1.0])
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        grid = P1C0Grid2D(x, y, X + Y)
        start = jnp.array([0.0, 0.0])
        end = jnp.array([1.0, 0.0])

        result = grid.segment_grid(start, end)

        npt.assert_allclose(result(jnp.array([0.0])), 0.0, atol=1e-9)
        npt.assert_allclose(result(jnp.array([1e-7])), 1e-7, atol=1e-9)
        npt.assert_allclose(result(jnp.array([1.0])), 1.0, atol=1e-6)

    def test_not_available_for_1d(self):
        self.assertFalse(
            hasattr(P1C0Grid1D(jnp.array([0.0, 1.0]), jnp.zeros(2)), "segment_grid")
        )

    def test_degenerate_segment_raises(self):
        grid = P1C0Grid2D(
            jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 1.0, 2.0]), jnp.zeros((3, 3))
        )
        with self.assertRaises(Exception):
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


class C0GridExtremaTest(unittest.TestCase):
    def test_available_in_every_dimension(self):
        self.assertTrue(
            hasattr(C0Grid1D(jnp.array([0.0, 1.0]), jnp.zeros(3), degree=2), "minimize")
        )
        self.assertTrue(
            hasattr(P1C0Grid1D(jnp.array([0.0, 1.0]), jnp.zeros(2)), "maximize")
        )
        grid4d = P2C0Grid4D(*([jnp.array([0.0, 1.0])] * 4), jnp.zeros((3, 3, 3, 3)))
        self.assertTrue(hasattr(grid4d, "minimize"))
        self.assertTrue(hasattr(grid4d, "maximize"))

    def test_p1_1d_monotonic_function_bounds_are_the_endpoints(self):
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        grid = P1C0Grid1D(x, x)

        minimum = grid.minimize()
        maximum = grid.maximize()

        npt.assert_allclose(minimum.f, 0.0, atol=1e-6)
        npt.assert_allclose(minimum.x, jnp.array([0.0]), atol=1e-6)
        npt.assert_allclose(maximum.f, 3.0, atol=1e-6)
        npt.assert_allclose(maximum.x, jnp.array([3.0]), atol=1e-6)
        npt.assert_allclose(grid(minimum.x), minimum.f, atol=1e-6)
        npt.assert_allclose(grid(maximum.x), maximum.f, atol=1e-6)
        # x=0 is uniquely the minimum, only in cell 0; x=3 is uniquely the
        # maximum, only in the last cell (2).
        npt.assert_array_equal(minimum.cell, jnp.array([0]))
        npt.assert_array_equal(maximum.cell, jnp.array([2]))

    def test_matches_module_level_minimize_and_maximize_on_a_single_cell_grid(self):
        a, b = 0.3, 0.4
        f_u = jnp.array([a**2, a**2 - a, (1 - a) ** 2])
        f_v = jnp.array([b**2, b**2 - b, (1 - b) ** 2])
        coefficients = f_u[:, None] + f_v[None, :]
        x = jnp.array([0.0, 1.0])
        y = jnp.array([0.0, 1.0])
        grid = P2C0Grid2D(x, y, coefficients)

        expected_min = minimize(Bernstein2D(coefficients), max_steps=200, eps=1e-8)
        expected_max = maximize(Bernstein2D(coefficients), max_steps=200, eps=1e-8)

        result_min = grid.minimize(max_steps=200, eps=1e-8)
        result_max = grid.maximize(max_steps=200, eps=1e-8)

        npt.assert_allclose(result_min.f, expected_min.f, atol=1e-6)
        npt.assert_allclose(result_min.x, expected_min.x, atol=1e-3)
        npt.assert_allclose(result_max.f, expected_max.f, atol=1e-6)
        npt.assert_allclose(result_max.x, expected_max.x, atol=1e-3)
        npt.assert_allclose(grid(result_min.x), result_min.f, rtol=1e-5, atol=1e-6)
        npt.assert_allclose(grid(result_max.x), result_max.f, rtol=1e-5, atol=1e-6)
        # Only one cell exists, so it must be reported both times.
        npt.assert_array_equal(result_min.cell, jnp.array([0, 0]))
        npt.assert_array_equal(result_max.cell, jnp.array([0, 0]))

    def test_finds_extrema_hidden_in_non_corner_cells(self):
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        y = jnp.array([0.0, 1.0, 2.0, 3.0])
        f = jnp.zeros((4, 4)).at[1, 2].set(10.0).at[2, 1].set(-5.0)
        grid = P1C0Grid2D(x, y, f)

        maximum = grid.maximize(max_steps=200, eps=1e-8)
        minimum = grid.minimize(max_steps=200, eps=1e-8)

        npt.assert_allclose(maximum.f, 10.0, atol=1e-5)
        npt.assert_allclose(maximum.x, jnp.array([1.0, 2.0]), atol=1e-3)
        npt.assert_allclose(minimum.f, -5.0, atol=1e-5)
        npt.assert_allclose(minimum.x, jnp.array([2.0, 1.0]), atol=1e-3)
        npt.assert_allclose(grid(maximum.x), maximum.f, atol=1e-5)
        npt.assert_allclose(grid(minimum.x), minimum.f, atol=1e-5)
        # Degree 1 means the extremum sits exactly on a shared vertex, so
        # any of the (up to 4) cells touching it is a legitimate answer —
        # check membership in that set rather than pinning one cell.
        self.assertIn(tuple(maximum.cell.tolist()), {(0, 1), (0, 2), (1, 1), (1, 2)})
        self.assertIn(tuple(minimum.cell.tolist()), {(1, 0), (1, 1), (2, 0), (2, 1)})

    def test_reports_the_unique_interior_cell_containing_the_minimum(self):
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        y = jnp.array([0.0, 1.0, 2.0, 3.0])
        a, b = 0.3, 0.4
        f_u = jnp.array([a**2, a**2 - a, (1 - a) ** 2])
        f_v = jnp.array([b**2, b**2 - b, (1 - b) ** 2])
        bump = f_u[:, None] + f_v[None, :]
        # A large baseline everywhere except the degree-2 window belonging
        # to cell (1, 1), where the true interior minimum is exactly 0 at
        # local (a, b) — verified separately to be far below every
        # neighboring cell's own minimum (their windows only ever see the
        # bump's boundary values, whose own minimum along that edge is
        # ~0.09, well above 0).
        f = jnp.full((7, 7), 1000.0).at[2:5, 2:5].set(bump)
        grid = P2C0Grid2D(x, y, f)

        result = grid.minimize(max_steps=200, eps=1e-8)

        npt.assert_allclose(result.f, 0.0, atol=1e-5)
        npt.assert_array_equal(result.cell, jnp.array([1, 1]))
        npt.assert_allclose(grid(result.x), result.f, atol=1e-5)

    def test_supports_a_leading_batch_dimension(self):
        x = jnp.array([0.0, 1.0, 2.0])
        f = jnp.stack([jnp.array([0.0, 1.0, 2.0]), jnp.array([5.0, 3.0, 0.0])])
        grid = P1C0Grid1D(x, f)

        minimum = grid.minimize(max_steps=100, eps=1e-7)
        maximum = grid.maximize(max_steps=100, eps=1e-7)

        self.assertEqual(minimum.f.shape, (2,))
        self.assertEqual(minimum.x.shape, (2, 1))
        self.assertEqual(minimum.cell.shape, (2, 1))
        npt.assert_allclose(minimum.f, jnp.array([0.0, 0.0]), atol=1e-6)
        npt.assert_allclose(minimum.x[:, 0], jnp.array([0.0, 2.0]), atol=1e-6)
        npt.assert_allclose(maximum.f, jnp.array([2.0, 5.0]), atol=1e-6)
        npt.assert_allclose(maximum.x[:, 0], jnp.array([2.0, 0.0]), atol=1e-6)
        # f0=[0,1,2] is increasing: its min is uniquely cell 0, max cell 1.
        # f1=[5,3,0] is decreasing: its min is uniquely cell 1, max cell 0.
        npt.assert_array_equal(minimum.cell[:, 0], jnp.array([0, 1]))
        npt.assert_array_equal(maximum.cell[:, 0], jnp.array([1, 0]))


if __name__ == "__main__":
    unittest.main()
