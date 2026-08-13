import unittest

import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import PiecewiseRationalCurve3D, RationalBernstein


def _endpoint_geometry(curve):
    first = curve.deriv()
    second = first.deriv()
    velocities = (first(0.0), first(1.0))
    tangents = tuple(velocity / jnp.linalg.vector_norm(velocity) for velocity in velocities)
    curvatures = tuple(
        (acceleration - jnp.dot(acceleration, tangent) * tangent)
        / jnp.dot(velocity, velocity)
        for velocity, acceleration, tangent in zip(
            velocities, (second(0.0), second(1.0)), tangents
        )
    )
    return jnp.stack(tangents), jnp.stack(curvatures)


def _source_curve():
    values = jnp.array(
        [[0.0, 0.5, 1.2, 2.0], [0.0, 0.1, 0.8, 1.0], [0.0, 0.2, 0.4, 0.5]]
    )
    return RationalBernstein(values, jnp.array([1.0, 0.8, 1.3, 1.0]))


class PiecewiseRationalCurve3DTest(unittest.TestCase):
    def test_interpolant_recovers_a_known_rational_curve(self):
        source = _source_curve()
        tangents, curvatures = _endpoint_geometry(source)
        curve = PiecewiseRationalCurve3D(
            jnp.stack((source(0.0), source(1.0))), tangents, curvatures
        )

        actual = curve.interpolant(0)

        self.assertEqual(curve.segment_count, 1)
        self.assertIsInstance(actual, RationalBernstein)
        self.assertEqual(actual.order, 3)
        npt.assert_allclose(actual.values, source.values, rtol=2e-4, atol=2e-4)
        npt.assert_allclose(actual.weights, source.weights, rtol=2e-4, atol=2e-4)

    def test_shared_nodes_reproduce_g2_data_for_every_segment(self):
        source = _source_curve()
        tangents, curvatures = _endpoint_geometry(source)
        positions = jnp.stack((source(0.0), source(1.0), source(1.0) + jnp.array([2.0, 0.0, 0.0])))
        straight_tangent = jnp.array([1.0, 0.0, 0.0])
        curve = PiecewiseRationalCurve3D(
            positions,
            jnp.stack((tangents[0], tangents[1], straight_tangent)),
            jnp.stack((curvatures[0], curvatures[1], jnp.zeros(3))),
        )

        first = curve.interpolant(0)
        npt.assert_allclose(first(0.0), positions[0], atol=2e-4)
        npt.assert_allclose(first(1.0), positions[1], atol=2e-4)

    def test_projects_curvature_and_translates_positions_only(self):
        curve = PiecewiseRationalCurve3D(
            jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            jnp.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            jnp.array([[5.0, 1.0, 0.0], [2.0, 0.0, 0.0]]),
        )
        offset = jnp.array([1.0, -2.0, 3.0])
        translated = curve.translated(offset)

        npt.assert_allclose(curve.tangents, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        npt.assert_allclose(curve.curvatures, [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        npt.assert_allclose(translated.positions, curve.positions + offset)
        npt.assert_allclose(translated.tangents, curve.tangents)
        npt.assert_allclose(translated.curvatures, curve.curvatures)

    def test_rejects_bad_shapes_and_unrepresentable_segments(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            PiecewiseRationalCurve3D(jnp.zeros((2, 2)), jnp.zeros((2, 3)), jnp.zeros((2, 3)))

        line = PiecewiseRationalCurve3D(
            jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            jnp.zeros((2, 3)),
        )
        self.assertRaisesRegex(Exception, "outside the curve", lambda: line.interpolant(1))


if __name__ == "__main__":
    unittest.main()
