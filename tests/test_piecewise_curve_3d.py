import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

from xbernstein import Bernstein, PiecewiseCurve3D


class PiecewiseCurve3DTest(unittest.TestCase):
    def test_interpolates_positions_and_tangents(self):
        positions = jnp.array([[0.0, 1.0, -1.0], [2.0, 3.0, 1.0], [4.0, 2.0, 5.0]])
        tangents = jnp.array([[1.0, 2.0, 0.0], [2.0, -1.0, 3.0], [0.0, 1.0, 2.0]])
        curve = PiecewiseCurve3D(positions, tangents)

        self.assertEqual(curve.segment_count, 2)
        self.assertIsInstance(curve.interpolant(0), Bernstein)
        for index in range(curve.segment_count):
            segment = curve.interpolant(index)
            npt.assert_allclose(segment(0.0), positions[index])
            npt.assert_allclose(segment(1.0), positions[index + 1])
            npt.assert_allclose(segment.deriv()(0.0), tangents[index], atol=1e-6)
            npt.assert_allclose(segment.deriv()(1.0), tangents[index + 1], atol=1e-6)

    def test_global_evaluation_and_c1_join_are_jittable(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [3.0, 1.0, 2.0]])
        tangents = jnp.array([[1.0, 0.0, 0.0], [2.0, -1.0, 1.0], [0.0, 1.0, 2.0]])
        curve = PiecewiseCurve3D(positions, tangents)
        values = curve(jnp.array([0.0, 1.0, 2.0]))
        npt.assert_allclose(values, positions)
        npt.assert_allclose(jax.jit(lambda t: curve(t))(jnp.array([0.25, 1.5])), curve(jnp.array([0.25, 1.5])))
        left = curve.interpolant(0).deriv()(1.0)
        right = curve.interpolant(1).deriv()(0.0)
        npt.assert_allclose(left, right, atol=1e-6)
        npt.assert_allclose(curve(jnp.array([-1.0, 3.0])), jnp.stack((positions[0], positions[-1])))

    def test_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            PiecewiseCurve3D(jnp.zeros((2, 2)), jnp.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "at least two"):
            PiecewiseCurve3D(jnp.zeros((1, 3)), jnp.zeros((1, 3)))
        curve = PiecewiseCurve3D(jnp.zeros((2, 3)), jnp.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "scalar"):
            curve.interpolant(jnp.array([0]))


if __name__ == "__main__":
    unittest.main()
