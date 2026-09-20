import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import (
    Bernstein,
    Bernstein2D,
    Bernstein3D,
    Bernstein4D,
    HermiteGrid1D,
    HermiteGrid2D,
    HermiteGrid3D,
    HermiteGrid4D,
    P3C1Grid1D,
    P3C1Grid2D,
    P3C1Grid3D,
    P3C1Grid4D,
    P4C1Grid1D,
    P4C1Grid2D,
    P4C1Grid3D,
    P4C1Grid4D,
    P5C1Grid1D,
    P5C1Grid2D,
    P5C1Grid3D,
    P5C1Grid4D,
)

_POLYNOMIAL_TYPES = (Bernstein, Bernstein2D, Bernstein3D, Bernstein4D)
_C1_CLASSES = {
    3: (P3C1Grid1D, P3C1Grid2D, P3C1Grid3D, P3C1Grid4D),
    4: (P4C1Grid1D, P4C1Grid2D, P4C1Grid3D, P4C1Grid4D),
    5: (P5C1Grid1D, P5C1Grid2D, P5C1Grid3D, P5C1Grid4D),
}
_HERMITE_CLASSES = (HermiteGrid1D, HermiteGrid2D, HermiteGrid3D, HermiteGrid4D)


def _multi_indices(dimension, total):
    """Same ordering convention as ``xbernstein.hermite._multi_indices(dim, 2, total)``."""
    return sorted(
        (
            alpha
            for alpha in itertools.product(range(2), repeat=dimension)
            if sum(alpha) == total
        ),
        reverse=True,
    )


def _mixed_partial_fn(base_fn, axes):
    """Return a function computing the mixed partial derivative along ``axes``."""
    fn = base_fn
    for axis in axes:
        def next_fn(point, fn=fn, axis=axis):
            return jax.grad(fn)(point)[axis]

        fn = next_fn
    return fn


def _make_polynomial(dimension, degree):
    """Build a fixed, deterministic degree-``degree`` tensor Bernstein polynomial."""
    shape = (degree + 1,) * dimension
    size = 1
    for extent in shape:
        size *= extent
    coefficients = (jnp.arange(size, dtype=jnp.float32).reshape(shape) + 1.0) * 0.1
    return _POLYNOMIAL_TYPES[dimension - 1](coefficients)


def _vertex_groups(poly, dimension):
    """Compute HermiteGrid-style jet groups at the corners of a single unit cell."""

    def value_fn(point):
        return poly(*point)

    vertices = list(itertools.product((0.0, 1.0), repeat=dimension))
    grid_shape = (2,) * dimension
    values = jnp.array([value_fn(jnp.array(v)) for v in vertices]).reshape(grid_shape)

    groups = [values]
    for total in range(1, dimension + 1):
        alphas = _multi_indices(dimension, total)
        rows = []
        for vertex in vertices:
            row = []
            for alpha in alphas:
                axes = tuple(a for a, order in enumerate(alpha) if order == 1)
                row.append(_mixed_partial_fn(value_fn, axes)(jnp.array(vertex)))
            rows.append(row)
        entries = jnp.array(rows).reshape(grid_shape + (len(alphas),))
        groups.append(entries[..., 0] if len(alphas) == 1 else entries)
    return tuple(groups)


def _grid_groups(fn, axes, dimension):
    """Compute HermiteGrid-style jet groups for ``fn`` over a full rectilinear grid."""

    def value_fn(point):
        return fn(*point)

    meshes = jnp.meshgrid(*axes, indexing="ij")
    grid_shape = tuple(axis.size for axis in axes)
    flat_points = jnp.stack([mesh.reshape(-1) for mesh in meshes], axis=-1)

    values = jax.vmap(value_fn)(flat_points).reshape(grid_shape)
    groups = [values]
    for total in range(1, dimension + 1):
        alphas = _multi_indices(dimension, total)

        def group_fn(point, alphas=alphas):
            outs = []
            for alpha in alphas:
                sub_axes = tuple(a for a, order in enumerate(alpha) if order == 1)
                outs.append(_mixed_partial_fn(value_fn, sub_axes)(point))
            return jnp.stack(outs)

        raw = jax.vmap(group_fn)(flat_points).reshape(grid_shape + (len(alphas),))
        groups.append(raw[..., 0] if len(alphas) == 1 else raw)
    return tuple(groups)


class P3EquivalenceTest(unittest.TestCase):
    """P3C1Grid must agree with the already-tested HermiteGrid (same construction)."""

    def test_matches_hermite_grid_in_every_dimension(self):
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                poly = _make_polynomial(dimension, degree=3)
                groups = _vertex_groups(poly, dimension)
                axes = (jnp.array([0.0, 1.0]),) * dimension

                c1_grid = _C1_CLASSES[3][dimension - 1](*axes, *groups)
                h_grid = _HERMITE_CLASSES[dimension - 1](*axes, *groups)
                index = jnp.zeros(dimension, dtype=jnp.int32)

                npt.assert_allclose(
                    c1_grid.cell_interpolant(index).c,
                    h_grid.cell_interpolant(index).c,
                    rtol=1e-5,
                    atol=1e-5,
                )
                npt.assert_allclose(
                    c1_grid.cell_interpolant(index).c, poly.c, rtol=1e-4, atol=1e-4
                )


class HigherDegreeReproductionTest(unittest.TestCase):
    """P4/P5 grids must exactly reproduce a polynomial from its own jets + interior data."""

    def test_reproduces_a_known_polynomial(self):
        cases = ((1, 4), (2, 4), (1, 5), (3, 5))
        for dimension, degree in cases:
            with self.subTest(dimension=dimension, degree=degree):
                poly = _make_polynomial(dimension, degree)
                groups = _vertex_groups(poly, dimension)
                axes = (jnp.array([0.0, 1.0]),) * dimension
                cls = _C1_CLASSES[degree][dimension - 1]

                grid = cls(*axes, *groups, poly.c)
                index = jnp.zeros(dimension, dtype=jnp.int32)

                self.assertEqual(grid.dof_shape, poly.c.shape)
                npt.assert_allclose(
                    grid.cell_interpolant(index).c, poly.c, rtol=1e-4, atol=1e-4
                )


class FaceContinuityTest(unittest.TestCase):
    """Neighboring cells must agree through first derivative across the crossed axis,
    and generically disagree one order beyond that — even on a nonuniform grid."""

    def _check_2d(self, degree):
        x = jnp.array([0.0, 1.0, 2.5])
        y = jnp.array([0.0, 1.0])

        def fn(a, b):
            return jnp.sin(a) + jnp.cos(b) + 0.3 * a * b**2

        groups = _grid_groups(fn, (x, y), 2)
        dof_shape = (degree * 2 + 1, degree * 1 + 1)
        interior = jnp.zeros(dof_shape)

        grid = _C1_CLASSES[degree][1](x, y, *groups, interior)
        left = grid.cell_interpolant(jnp.array([0, 0]))
        right = grid.cell_interpolant(jnp.array([1, 0]))
        width_left = x[1] - x[0]
        width_right = x[2] - x[1]
        v = jnp.linspace(0.0, 1.0, 5)

        for order in (0, 1):
            left_value = left.deriv(order, axis=0)(jnp.ones_like(v), v) / width_left**order
            right_value = right.deriv(order, axis=0)(jnp.zeros_like(v), v) / width_right**order
            npt.assert_allclose(left_value, right_value, rtol=1e-4, atol=1e-4)

        next_order = 2
        left_next = left.deriv(next_order, axis=0)(jnp.ones_like(v), v) / width_left**next_order
        right_next = right.deriv(next_order, axis=0)(jnp.zeros_like(v), v) / width_right**next_order
        self.assertFalse(
            bool(jnp.allclose(left_next, right_next, rtol=1e-4, atol=1e-4)),
            "derivative one order beyond C1 should generically differ",
        )

    def test_p4_grid_2d(self):
        self._check_2d(4)

    def test_p5_grid_2d(self):
        self._check_2d(5)

    def test_p4_grid_3d_shares_the_entire_face_between_two_cells(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0])
        z = jnp.array([0.0, 1.0])

        def fn(a, b, c):
            return jnp.sin(a) + b * c + 0.2 * a * b

        groups = _grid_groups(fn, (x, y, z), 3)
        interior = jnp.zeros((4 * 2 + 1, 4 * 1 + 1, 4 * 1 + 1))
        grid = P4C1Grid3D(x, y, z, *groups, interior)

        left = grid.cell_interpolant(jnp.array([0, 0, 0]))
        right = grid.cell_interpolant(jnp.array([1, 0, 0]))
        npt.assert_array_equal(left.c[-1], right.c[0])


class InteriorDataPlacementTest(unittest.TestCase):
    def test_interior_positions_use_supplied_data_boundary_stays_jet_determined(self):
        x = jnp.array([0.0, 1.0])
        f = jnp.zeros(2)
        d1 = jnp.zeros(2)
        interior = jnp.zeros(6).at[2].set(7.0).at[3].set(-3.0)

        grid = P5C1Grid1D(x, f, d1, interior)
        c = grid.cell_interpolant(jnp.array([0])).c

        npt.assert_allclose(c[:2], jnp.zeros(2))
        npt.assert_allclose(c[2], 7.0)
        npt.assert_allclose(c[3], -3.0)
        npt.assert_allclose(c[4:], jnp.zeros(2))


class ValidationAndSmokeTest(unittest.TestCase):
    def test_missing_required_interior_coefficients_raises(self):
        x = jnp.array([0.0, 1.0])
        with self.assertRaises(TypeError):
            P4C1Grid1D(x, jnp.zeros(2), jnp.zeros(2), None)

    def test_wrong_jet_shape_raises(self):
        x = jnp.array([0.0, 1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "d1 must have shape"):
            P3C1Grid1D(x, jnp.zeros(3), jnp.zeros(2))

    def test_wrong_interior_coefficients_shape_raises(self):
        x = jnp.array([0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "interior_coefficients must have shape"):
            P4C1Grid1D(x, jnp.zeros(2), jnp.zeros(2), jnp.zeros(3))

    def test_split_segment_is_available_only_from_2d(self):
        grid = P3C1Grid2D(
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
                P3C1Grid1D(jnp.array([0.0, 1.0]), jnp.zeros(2), jnp.zeros(2)),
                "split_segment",
            )
        )

    def test_3d_and_4d_construct_and_evaluate(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0])
        z = jnp.array([0.0, 1.0])
        w = jnp.array([0.0, 1.0])

        grid3d = P4C1Grid3D(
            x,
            y,
            z,
            jnp.zeros((3, 2, 2)),
            jnp.zeros((3, 2, 2, 3)),
            jnp.zeros((3, 2, 2, 3)),
            jnp.zeros((3, 2, 2)),
            jnp.zeros((9, 5, 5)),
        )
        self.assertEqual(grid3d.dof_shape, (9, 5, 5))
        npt.assert_allclose(grid3d(jnp.array([0.5, 0.5, 0.5])), 0.0)

        grid4d = P4C1Grid4D(
            x,
            y,
            z,
            w,
            jnp.zeros((3, 2, 2, 2)),
            jnp.zeros((3, 2, 2, 2, 4)),
            jnp.zeros((3, 2, 2, 2, 6)),
            jnp.zeros((3, 2, 2, 2, 4)),
            jnp.zeros((3, 2, 2, 2)),
            jnp.zeros((9, 5, 5, 5)),
        )
        self.assertEqual(grid4d.dof_shape, (9, 5, 5, 5))
        npt.assert_allclose(grid4d(jnp.array([0.5, 0.5, 0.5, 0.5])), 0.0)


if __name__ == "__main__":
    unittest.main()
