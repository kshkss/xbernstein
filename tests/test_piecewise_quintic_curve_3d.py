import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt
from jaxtyping import TypeCheckError

from xbernstein import Bernstein, P5C1, PiecewiseQuinticCurve3D


def _geometry(curve, parameter):
    """Return (position, velocity, curvature vector, torsion) at ``parameter``."""
    velocity = jax.jacfwd(curve)(parameter)
    acceleration = jax.jacfwd(jax.jacfwd(curve))(parameter)
    jerk = jax.jacfwd(jax.jacfwd(jax.jacfwd(curve)))(parameter)
    tangent = velocity / jnp.linalg.vector_norm(velocity)
    curvature = (
        acceleration - jnp.dot(acceleration, tangent) * tangent
    ) / jnp.dot(velocity, velocity)
    binormal = jnp.cross(velocity, acceleration)
    torsion = jnp.dot(binormal, jerk) / jnp.dot(binormal, binormal)
    return curve(parameter), velocity, curvature, torsion


def _first_source_curve():
    coefficients = jnp.array(
        [
            [0.0, 0.3, 0.9, 1.3, 1.6, 2.0],
            [0.0, 0.2, -0.1, 0.4, 0.9, 1.0],
            [0.0, 0.5, 1.1, 0.7, 0.3, 0.6],
        ]
    )
    return Bernstein(coefficients)


def _second_source_curve(p0, v0):
    c0 = p0
    c1 = p0 + v0 / 5.0
    c2 = jnp.array([1.6, -0.3, 0.4])
    c3 = jnp.array([2.0, 0.1, -0.2])
    c4 = jnp.array([2.3, 0.4, 0.1])
    c5 = jnp.array([2.6, 0.6, 0.3])
    return Bernstein(jnp.stack([c0, c1, c2, c3, c4, c5], axis=-1))


class PiecewiseQuinticCurve3DTest(unittest.TestCase):
    def test_reproduces_a_known_quintic_curve(self):
        source = _first_source_curve()
        p0, v0, k0, t0 = _geometry(source, jnp.array(0.0))
        p1, v1, k1, t1 = _geometry(source, jnp.array(1.0))

        curve = PiecewiseQuinticCurve3D(p0, v0, p1, v1, k0, t0, k1, t1)
        element = curve.to_p5c1()
        actual = element.cell_interpolant(0)

        self.assertIsInstance(element, P5C1)
        self.assertEqual(curve.node_count, 2)
        self.assertEqual(curve.segment_count, 1)
        self.assertEqual(actual.order, 5)
        npt.assert_allclose(actual.c, source.c, atol=1e-4, rtol=1e-4)

        parameters = jnp.linspace(0.0, 1.0, 5)
        npt.assert_allclose(
            curve(parameters), jax.vmap(source)(parameters), atol=1e-4
        )

    def test_append_adds_a_segment_without_mutating_the_original(self):
        source1 = _first_source_curve()
        p0, v0, k0, t0 = _geometry(source1, jnp.array(0.0))
        p1, v1, k1, t1 = _geometry(source1, jnp.array(1.0))
        curve = PiecewiseQuinticCurve3D(p0, v0, p1, v1, k0, t0, k1, t1)

        source2 = _second_source_curve(p1, v1)
        p1_again, v1_again, k1_start, t1_start = _geometry(source2, jnp.array(0.0))
        p2, v2, k2, t2 = _geometry(source2, jnp.array(1.0))
        npt.assert_allclose(p1_again, p1, atol=1e-6)
        npt.assert_allclose(v1_again, v1, atol=1e-6)

        grown = curve.append(p2, v2, k1_start, t1_start, k2, t2)

        self.assertEqual(curve.segment_count, 1)  # original left untouched
        self.assertEqual(grown.segment_count, 2)
        self.assertEqual(grown.node_count, 3)

        element = grown.to_p5c1()
        first_cell = element.cell_interpolant(0)
        second_cell = element.cell_interpolant(1)
        npt.assert_allclose(first_cell.c, source1.c, atol=1e-4, rtol=1e-4)
        npt.assert_allclose(second_cell.c, source2.c, atol=1e-4, rtol=1e-4)

        # position/velocity continuity across the shared node.
        npt.assert_allclose(
            first_cell(1.0), second_cell(0.0), atol=1e-4
        )
        npt.assert_allclose(
            first_cell.deriv()(1.0), second_cell.deriv()(0.0), atol=1e-4
        )

    def test_zero_curvature_endpoint_raises(self):
        p0 = jnp.zeros(3)
        v0 = jnp.array([1.0, 0.0, 0.0])
        p1 = jnp.array([1.0, 0.0, 0.0])
        v1 = jnp.array([1.0, 0.0, 0.0])
        curvature = jnp.zeros(3)
        torsion = jnp.array(0.0)
        curve = PiecewiseQuinticCurve3D(
            p0, v0, p1, v1, curvature, torsion, curvature, torsion
        )
        with self.assertRaises(Exception):
            curve.to_p5c1()

    def test_rejects_bad_shapes(self):
        with self.assertRaises(TypeCheckError):
            PiecewiseQuinticCurve3D(
                jnp.zeros(2),
                jnp.zeros(3),
                jnp.zeros(3),
                jnp.zeros(3),
                jnp.zeros(3),
                jnp.array(0.0),
                jnp.zeros(3),
                jnp.array(0.0),
            )


if __name__ == "__main__":
    unittest.main()
