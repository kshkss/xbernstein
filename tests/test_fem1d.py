import math
import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import (
    P1C0,
    P2C0,
    P3C0,
    P3C1,
    P4C1,
    P5C1,
    P5C2,
    hermite_interpolate_1d,
    linear_interpolate_1d,
    quintic_hermite_interpolate_1d,
)

# (class, degree, continuity, needs_interior_coefficients)
_CASES = (
    (P1C0, 1, 0, False),
    (P2C0, 2, 0, True),
    (P3C0, 3, 0, True),
    (P3C1, 3, 1, False),
    (P4C1, 4, 1, True),
    (P5C1, 5, 1, True),
    (P5C2, 5, 2, False),
)


def _prod(shape):
    result = 1
    for size in shape:
        result *= size
    return result


def _construct(cls, nodes, connectivity, jets, interior_coefficients):
    if interior_coefficients is None:
        return cls(nodes, connectivity, *jets)
    return cls(nodes, connectivity, *jets, interior_coefficients)


def _synthetic_instance(cls, degree, continuity, needs_interior, nodes, connectivity, value_shape):
    """Build an instance with distinctive, non-degenerate synthetic DOF data."""
    jet_size = continuity + 1
    node_count = nodes.shape[0]
    cell_count = connectivity.shape[0]
    node_shape = (node_count,) + value_shape
    base = float(degree * 10 + continuity)

    n = _prod(node_shape)
    f = jnp.arange(n, dtype=jnp.float32).reshape(node_shape) * 0.1 + base
    jets = [f]
    if jet_size > 1:
        jets.append(jnp.arange(n, dtype=jnp.float32).reshape(node_shape) * 0.05 + 1.0)
    if jet_size > 2:
        jets.append(jnp.arange(n, dtype=jnp.float32).reshape(node_shape) * 0.02 - 0.5)

    interior_coefficients = None
    if needs_interior:
        interior_dof_count = degree + 1 - 2 * jet_size
        cell_shape = (cell_count,) + value_shape + (interior_dof_count,)
        interior_coefficients = (
            jnp.arange(_prod(cell_shape), dtype=jnp.float32).reshape(cell_shape) * 0.3 - 1.0
        )

    return _construct(cls, nodes, connectivity, jets, interior_coefficients)


class ExactCaseTest(unittest.TestCase):
    """P1C0/P3C1/P5C2 are the exact (interior-dof-free) cases: for a single
    cell they must reproduce the existing hermite.py functions exactly."""

    def test_p1c0_matches_linear_interpolate_1d(self):
        nodes = jnp.array([0.0, 2.0])
        connectivity = jnp.array([[0, 1]])
        f = jnp.array([1.0, 5.0])
        grid = P1C0(nodes, connectivity, f)
        expected = linear_interpolate_1d(f)
        npt.assert_allclose(grid.cell_interpolant(0).c, expected.c, rtol=1e-6)

    def test_p3c1_matches_hermite_interpolate_1d(self):
        nodes = jnp.array([0.0, 2.0])
        connectivity = jnp.array([[0, 1]])
        width = 2.0
        f = jnp.array([1.0, 5.0])
        d1 = jnp.array([2.0, -1.0])
        grid = P3C1(nodes, connectivity, f, d1)
        expected = hermite_interpolate_1d(f, d1 * width)
        npt.assert_allclose(grid.cell_interpolant(0).c, expected.c, rtol=1e-6)

    def test_p5c2_matches_quintic_hermite_interpolate_1d(self):
        nodes = jnp.array([0.0, 2.0])
        connectivity = jnp.array([[0, 1]])
        width = 2.0
        f = jnp.array([1.0, 5.0])
        d1 = jnp.array([2.0, -1.0])
        d2 = jnp.array([0.5, -0.3])
        grid = P5C2(nodes, connectivity, f, d1, d2)
        expected = quintic_hermite_interpolate_1d(f, d1 * width, d2 * width**2)
        npt.assert_allclose(grid.cell_interpolant(0).c, expected.c, rtol=1e-6)


class ContinuityTest(unittest.TestCase):
    """Neighboring cells must agree through derivative order `continuity`
    at their shared node, and generically disagree one order beyond that."""

    def test_shared_node_continuity_and_next_order_mismatch(self):
        nodes = jnp.array([0.0, 1.0, 2.5])
        connectivity = jnp.array([[0, 1], [1, 2]])
        for cls, degree, continuity, needs_interior in _CASES:
            for value_shape in ((), (3,)):
                with self.subTest(cls=cls.__name__, value_shape=value_shape):
                    grid = _synthetic_instance(
                        cls, degree, continuity, needs_interior, nodes, connectivity, value_shape
                    )
                    left = grid.cell_interpolant(0)
                    right = grid.cell_interpolant(1)
                    width_left = nodes[1] - nodes[0]
                    width_right = nodes[2] - nodes[1]
                    for order in range(continuity + 1):
                        left_value = left.deriv(order)(1.0) / width_left**order
                        right_value = right.deriv(order)(0.0) / width_right**order
                        npt.assert_allclose(left_value, right_value, rtol=1e-4, atol=1e-4)
                    next_order = continuity + 1
                    left_next = left.deriv(next_order)(1.0) / width_left**next_order
                    right_next = right.deriv(next_order)(0.0) / width_right**next_order
                    self.assertFalse(
                        bool(jnp.allclose(left_next, right_next, rtol=1e-4, atol=1e-4)),
                        "derivative one order beyond the continuity class should generically differ",
                    )


class ConnectivityOrderIndependenceTest(unittest.TestCase):
    def test_permuted_node_storage_gives_identical_results(self):
        nodes = jnp.array([0.0, 1.0, 2.5])
        connectivity = jnp.array([[0, 1], [1, 2]])
        for cls, degree, continuity, needs_interior in _CASES:
            with self.subTest(cls=cls.__name__):
                grid = _synthetic_instance(
                    cls, degree, continuity, needs_interior, nodes, connectivity, ()
                )

                permutation = jnp.array([2, 0, 1])
                inverse = jnp.zeros(3, dtype=jnp.int32).at[permutation].set(jnp.arange(3))
                nodes_permuted = nodes[permutation]
                connectivity_permuted = inverse[connectivity]

                jets_permuted = [jet[permutation] for jet in grid._jets]
                permuted = _construct(
                    cls,
                    nodes_permuted,
                    connectivity_permuted,
                    jets_permuted,
                    grid.interior_coefficients if needs_interior else None,
                )

                points = jnp.array([0.2, 0.8, 1.5, 2.3])
                npt.assert_allclose(grid(points), permuted(points), rtol=1e-6)


class VectorValuedTest(unittest.TestCase):
    def test_call_shape_and_reproduction(self):
        nodes = jnp.array([0.0, 1.0, 2.5])
        connectivity = jnp.array([[0, 1], [1, 2]])
        value_shape = (3,)
        for cls, degree, continuity, needs_interior in _CASES:
            with self.subTest(cls=cls.__name__):
                grid = _synthetic_instance(
                    cls, degree, continuity, needs_interior, nodes, connectivity, value_shape
                )
                self.assertEqual(grid(jnp.array(0.5)).shape, value_shape)
                points = jnp.array([0.2, 0.8, 1.5, 2.3])
                self.assertEqual(grid(points).shape, points.shape + value_shape)
                npt.assert_allclose(grid(nodes[0]), grid.f[0], rtol=1e-5, atol=1e-5)
                npt.assert_allclose(grid(nodes[-1]), grid.f[-1], rtol=1e-5, atol=1e-5)


class PolynomialReproductionTest(unittest.TestCase):
    """Affine node data with matching interior control points must reproduce
    an affine function exactly, scalar- and vector-valued."""

    def test_affine_reproduction(self):
        nodes = jnp.array([0.0, 1.0, 2.5, 4.0])
        connectivity = jnp.array([[0, 1], [1, 2], [2, 3]])

        for value_shape in ((), (2,)):
            slope = jnp.full(value_shape, 1.5) if value_shape else jnp.array(1.5)
            intercept = jnp.full(value_shape, -0.5) if value_shape else jnp.array(-0.5)

            def affine(x):
                return intercept + slope * x

            f = jnp.stack([affine(x) for x in nodes], axis=0)
            zero_deriv = jnp.zeros_like(f)

            for cls, degree, continuity, needs_interior in _CASES:
                with self.subTest(cls=cls.__name__, value_shape=value_shape):
                    jet_size = continuity + 1
                    jets = [f]
                    if jet_size > 1:
                        d1 = jnp.stack([slope for _ in nodes], axis=0)
                        jets.append(d1)
                    if jet_size > 2:
                        jets.append(zero_deriv)

                    interior_coefficients = None
                    if needs_interior:
                        interior_dof_count = degree + 1 - 2 * jet_size
                        cells = []
                        for cell in range(connectivity.shape[0]):
                            i, j = connectivity[cell]
                            x0, x1 = nodes[i], nodes[j]
                            # Bezier control points of an affine function are
                            # exactly the function sampled at t = i / degree;
                            # interior index k corresponds to Bezier index
                            # jet_size + k.
                            control_points = [
                                affine(x0 + (jet_size + k) / degree * (x1 - x0))
                                for k in range(interior_dof_count)
                            ]
                            cells.append(jnp.stack(control_points, axis=-1))
                        interior_coefficients = jnp.stack(cells, axis=0)

                    grid = _construct(cls, nodes, connectivity, jets, interior_coefficients)
                    sample_points = jnp.array([0.0, 0.3, 1.0, 1.7, 2.5, 3.2, 4.0])
                    expected = jnp.stack([affine(x) for x in sample_points], axis=0)
                    npt.assert_allclose(grid(sample_points), expected, rtol=1e-4, atol=1e-4)


class DomainClippingTest(unittest.TestCase):
    def test_points_outside_domain_clip_to_boundary(self):
        nodes = jnp.array([0.0, 1.0, 2.5])
        connectivity = jnp.array([[0, 1], [1, 2]])
        f = jnp.array([0.0, 1.0, 3.0])
        grid = P1C0(nodes, connectivity, f)
        npt.assert_allclose(grid(jnp.array(-5.0)), grid(nodes[0]))
        npt.assert_allclose(grid(jnp.array(50.0)), grid(nodes[-1]))


class JitCompatibilityTest(unittest.TestCase):
    def test_jit_matches_eager(self):
        nodes = jnp.array([0.0, 1.0, 2.5])
        connectivity = jnp.array([[0, 1], [1, 2]])
        for cls, degree, continuity, needs_interior in _CASES:
            with self.subTest(cls=cls.__name__):
                grid = _synthetic_instance(
                    cls, degree, continuity, needs_interior, nodes, connectivity, ()
                )
                points = jnp.array([0.2, 0.8, 1.5, 2.3])
                jitted = jax.jit(lambda p, grid=grid: grid(p))
                npt.assert_allclose(grid(points), jitted(points), rtol=1e-6)


class ValidationTest(unittest.TestCase):
    def test_missing_required_interior_coefficients_raises(self):
        nodes = jnp.array([0.0, 1.0])
        connectivity = jnp.array([[0, 1]])
        f = jnp.array([0.0, 1.0])
        with self.assertRaises(TypeError):
            P2C0(nodes, connectivity, f, None)

    def test_interior_coefficients_rejected_when_not_needed(self):
        nodes = jnp.array([0.0, 1.0])
        connectivity = jnp.array([[0, 1]])
        f = jnp.array([0.0, 1.0])
        with self.assertRaises(TypeError):
            P1C0(nodes, connectivity, f, jnp.zeros((1, 0)))

    def test_wrong_f_shape_raises(self):
        nodes = jnp.array([0.0, 1.0, 2.0])
        connectivity = jnp.array([[0, 1], [1, 2]])
        with self.assertRaises(ValueError):
            P1C0(nodes, connectivity, jnp.array([0.0, 1.0]))

    def test_value_shape_mismatch_between_jets_raises(self):
        nodes = jnp.array([0.0, 1.0])
        connectivity = jnp.array([[0, 1]])
        f = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        d1 = jnp.array([1.0, 1.0])
        with self.assertRaises(ValueError):
            P3C1(nodes, connectivity, f, d1)

    def test_wrong_interior_coefficients_shape_raises(self):
        nodes = jnp.array([0.0, 1.0])
        connectivity = jnp.array([[0, 1]])
        f = jnp.array([0.0, 1.0])
        with self.assertRaises(ValueError):
            P2C0(nodes, connectivity, f, jnp.zeros((1, 2)))

    def test_out_of_range_connectivity_raises_at_runtime(self):
        nodes = jnp.array([0.0, 1.0])
        connectivity = jnp.array([[0, 5]])
        f = jnp.array([0.0, 1.0])
        with self.assertRaises(Exception):
            P1C0(nodes, connectivity, f)

    def test_non_positive_cell_width_raises_at_runtime(self):
        nodes = jnp.array([1.0, 0.5])
        connectivity = jnp.array([[0, 1]])
        f = jnp.array([0.0, 1.0])
        with self.assertRaises(Exception):
            P1C0(nodes, connectivity, f)


class FieldLayoutTest(unittest.TestCase):
    def test_node_dofs_are_stored_verbatim(self):
        nodes = jnp.array([0.0, 1.0, 2.5])
        connectivity = jnp.array([[0, 1], [1, 2]])
        f = jnp.array([0.0, 1.0, 3.0])
        d1 = jnp.array([1.0, -0.5, 2.0])
        interior = jnp.array([[0.3], [0.9]])
        grid = P4C1(nodes, connectivity, f, d1, interior)
        npt.assert_array_equal(grid.f, f)
        npt.assert_array_equal(grid.d1, d1)
        npt.assert_array_equal(grid.interior_coefficients, interior)


if __name__ == "__main__":
    unittest.main()
