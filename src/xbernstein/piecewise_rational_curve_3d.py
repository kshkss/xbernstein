r"""Piecewise C¹ and G² cubic rational Bernstein curves in three dimensions."""

import equinox as eqx
from beartype import beartype
import jax
import jax.numpy as jnp
from jaxtyping import Float, Shaped, jaxtyped

from .hermite import hermite_interpolate_1d
from .rational_bernstein import RationalBernstein, _from_homogeneous


def _orthogonal(
    vector: Float[jax.Array, "3"], tangent: Float[jax.Array, "3"]
) -> Float[jax.Array, "3"]:
    """Return the component of ``vector`` normal to the unit ``tangent``."""
    return vector - jnp.dot(vector, tangent) * tangent


def _solve_two_columns(
    matrix: Float[jax.Array, "3 2"], rhs: Float[jax.Array, "3"]
) -> tuple[Float[jax.Array, "2"], Float[jax.Array, "3"]]:
    """Solve a possibly overdetermined two-column system and return its residual."""
    solution = jnp.linalg.lstsq(matrix, rhs, rcond=None)[0]
    return solution, matrix @ solution - rhs


def _solve_segment(p0, p3, t0, t1, k0, k1) -> RationalBernstein:
    """Construct one positive-weight cubic rational G² segment."""
    chord = p3 - p0
    tolerance = 256.0 * jnp.finfo(p0.dtype).eps
    straight = (jnp.linalg.vector_norm(k0) <= tolerance) & (
        jnp.linalg.vector_norm(k1) <= tolerance
    )
    chord_direction = chord / jnp.linalg.vector_norm(chord)
    straight_valid = (
        (jnp.linalg.vector_norm(jnp.cross(t0, chord_direction)) <= tolerance)
        & (jnp.linalg.vector_norm(jnp.cross(t1, chord_direction)) <= tolerance)
        & (jnp.dot(t0, chord_direction) > 0.0)
        & (jnp.dot(t1, chord_direction) > 0.0)
    )

    def make_straight(_):
        values = jnp.stack((p0, p0 + chord / 3.0, p0 + 2.0 * chord / 3.0, p3), axis=-1)
        return values, jnp.ones(4, dtype=p0.dtype), jnp.array(False)

    def make_curved(_):
        start_matrix = jnp.stack((_orthogonal(t1, t0), k0), axis=-1)
        start_solution, start_residual = _solve_two_columns(
            start_matrix, _orthogonal(chord, t0)
        )
        beta, mu0 = start_solution
        end_matrix = jnp.stack((_orthogonal(t0, t1), -k1), axis=-1)
        end_solution, end_residual = _solve_two_columns(
            end_matrix, _orthogonal(chord, t1)
        )
        alpha, mu1 = end_solution
        valid = (
            (jnp.linalg.vector_norm(k0) > tolerance)
            & (jnp.linalg.vector_norm(k1) > tolerance)
            & (jnp.linalg.vector_norm(start_residual) <= tolerance)
            & (jnp.linalg.vector_norm(end_residual) <= tolerance)
            & (alpha > tolerance)
            & (beta > tolerance)
            & (mu0 > tolerance)
            & (mu1 > tolerance)
        )
        alpha_factor = 3.0 * alpha**2 / (2.0 * mu0)
        beta_factor = 3.0 * beta**2 / (2.0 * mu1)
        w1 = (1.0 / (beta_factor * alpha_factor**2)) ** (1.0 / 3.0)
        w2 = alpha_factor * w1**2
        weights = jnp.stack(
            (
                jnp.array(1.0, dtype=w1.dtype),
                w1,
                w2,
                jnp.array(1.0, dtype=w1.dtype),
            )
        )
        values = jnp.stack((p0, p0 + alpha * t0, p3 - beta * t1, p3), axis=-1)
        valid = valid & jnp.all(jnp.isfinite(weights)) & jnp.all(weights > 0.0)
        return values, weights, ~valid

    values, weights, invalid = jax.lax.cond(
        straight, make_straight, make_curved, operand=None
    )
    invalid = invalid | (straight & ~straight_valid)
    values = eqx.error_if(
        values,
        invalid,
        "endpoint data do not admit a positive-weight cubic rational G² interpolant",
    )
    return RationalBernstein(values, weights)


def _endpoint_speeds(
    controls: Float[jax.Array, "4 4"],
) -> tuple[Float[jax.Array, ""], Float[jax.Array, ""]]:
    """Return Euclidean speed magnitudes at both ends of homogeneous controls."""
    xyz = controls[:3]
    weights = controls[3]
    values = xyz / weights[None, :]
    start_velocity = 3.0 * weights[1] / weights[0] * (values[:, 1] - values[:, 0])
    end_velocity = 3.0 * weights[-2] / weights[-1] * (values[:, -1] - values[:, -2])
    return (
        jnp.linalg.vector_norm(start_velocity),
        jnp.linalg.vector_norm(end_velocity),
    )


@jaxtyped(typechecker=beartype)
class PiecewiseRationalCurve3D(eqx.Module):
    r"""Store homogeneous Hermite data for a C¹ chain of 3D G² rational cubics.

    The constructor accepts node ``positions``, unit directions ``tangents``,
    and curvature vectors with shape ``(node_count, 3)``. It solves every
    segment once and stores only endpoint values and derivatives of the four
    homogeneous components ``(X,Y,Z,W)``. Euclidean points are
    ``(X/W,Y/W,Z/W)``. The original geometric data are exposed as properties
    computed from the homogeneous representation.

    The first segment keeps its original parameterization and endpoint
    weights of one. Later segments are reparameterized by endpoint-preserving
    Mobius maps so Euclidean velocity is continuous at every integer knot.
    Homogeneous endpoint weights agree across each knot; the final weight is
    the positive value produced by the left-to-right propagation.
    """

    homogeneous_f: Float[jax.Array, "segments 4 2"]
    homogeneous_d1: Float[jax.Array, "segments 4 2"]

    def __init__(
        self,
        positions: Float[jax.Array, "nodes 3"],
        tangents: Float[jax.Array, "nodes 3"],
        curvatures: Float[jax.Array, "nodes 3"],
    ):
        groups = tuple(
            jnp.asarray(group) for group in (positions, tangents, curvatures)
        )
        if any(group.ndim != 2 or group.shape[1] != 3 for group in groups):
            raise ValueError(
                "positions, tangents, and curvatures must each have shape (node_count, 3)"
            )
        if len({group.shape[0] for group in groups}) != 1:
            raise ValueError(
                "positions, tangents, and curvatures must have equal node counts"
            )
        if groups[0].shape[0] < 2:
            raise ValueError("at least two nodes are required")

        dtype = jnp.result_type(*groups, jnp.float32)
        positions, tangents, curvatures = (group.astype(dtype) for group in groups)
        finite = (
            jnp.all(jnp.isfinite(positions))
            & jnp.all(jnp.isfinite(tangents))
            & jnp.all(jnp.isfinite(curvatures))
        )
        positions = eqx.error_if(
            positions, ~finite, "curve node data must contain only finite values"
        )
        tangent_norms = jnp.linalg.vector_norm(tangents, axis=-1)
        tangents = eqx.error_if(
            tangents,
            jnp.any(tangent_norms <= jnp.finfo(dtype).eps),
            "curve tangents must be non-zero",
        )
        tangents = tangents / tangent_norms[:, None]
        curvatures = (
            curvatures
            - jnp.sum(curvatures * tangents, axis=-1, keepdims=True) * tangents
        )
        segment_lengths = jnp.linalg.vector_norm(jnp.diff(positions, axis=0), axis=-1)
        positions = eqx.error_if(
            positions,
            jnp.any(segment_lengths <= jnp.finfo(dtype).eps),
            "curve segments must have distinct endpoints",
        )

        segment_controls = []
        endpoint_speeds = []
        for index in range(positions.shape[0] - 1):
            segment = _solve_segment(
                positions[index],
                positions[index + 1],
                tangents[index],
                tangents[index + 1],
                curvatures[index],
                curvatures[index + 1],
            )
            xyz = segment.h[..., 0, :]
            w = segment.h[0, 1, :]
            controls = jnp.concatenate((xyz, w[None, :]), axis=0)
            segment_controls.append(controls)
            endpoint_speeds.append(_endpoint_speeds(controls))

        values = []
        derivatives = []
        parameter_scale = jnp.array(1.0, dtype=dtype)
        node_weight = jnp.array(1.0, dtype=dtype)
        powers = jnp.arange(4, dtype=dtype)
        for index, controls in enumerate(segment_controls):
            if index:
                previous_end_speed = endpoint_speeds[index - 1][1]
                start_speed = endpoint_speeds[index][0]
                parameter_scale = previous_end_speed / (parameter_scale * start_speed)
            next_node_weight = node_weight * parameter_scale**3
            scales = node_weight * parameter_scale**powers
            invalid = (
                ~jnp.isfinite(parameter_scale)
                | (parameter_scale <= 0.0)
                | ~jnp.isfinite(node_weight)
                | (node_weight <= 0.0)
                | ~jnp.isfinite(next_node_weight)
                | (next_node_weight <= 0.0)
                | ~jnp.all(jnp.isfinite(scales))
                | jnp.any(scales <= 0.0)
            )
            parameter_scale = eqx.error_if(
                parameter_scale,
                invalid,
                "curve segments do not admit a finite positive C1 Mobius reparameterization",
            )
            controls = controls * scales[None, :]
            values.append(jnp.stack((controls[:, 0], controls[:, -1]), axis=-1))
            derivatives.append(
                3.0
                * jnp.stack(
                    (
                        controls[:, 1] - controls[:, 0],
                        controls[:, -1] - controls[:, -2],
                    ),
                    axis=-1,
                )
            )
            node_weight = next_node_weight
        self.homogeneous_f = jnp.stack(values)
        self.homogeneous_d1 = jnp.stack(derivatives)

    @classmethod
    def _from_hermite(cls, homogeneous_f, homogeneous_d1):
        result = object.__new__(cls)
        object.__setattr__(result, "homogeneous_f", homogeneous_f)
        object.__setattr__(result, "homogeneous_d1", homogeneous_d1)
        return result

    @property
    def segment_count(self) -> int:
        """Return the number of interpolable intervals."""
        return self.homogeneous_f.shape[0]

    @property
    def dtype(self) -> str:
        """Return the scalar dtype used by the homogeneous data."""
        return str(self.homogeneous_f.dtype)

    def _node_data(self):
        values = jnp.concatenate(
            (self.homogeneous_f[:1, :, 0], self.homogeneous_f[:, :, 1]), axis=0
        )
        derivatives = jnp.concatenate(
            (self.homogeneous_d1[:1, :, 0], self.homogeneous_d1[:, :, 1]), axis=0
        )
        return values, derivatives

    @property
    def positions(self) -> Float[jax.Array, "nodes 3"]:
        """Recover Euclidean node positions from ``(X,Y,Z,W)``."""
        values, _ = self._node_data()
        return values[:, :3] / values[:, 3, None]

    def _velocities(self):
        values, derivatives = self._node_data()
        xyz, w = values[:, :3], values[:, 3, None]
        dxyz, dw = derivatives[:, :3], derivatives[:, 3, None]
        return (dxyz * w - xyz * dw) / w**2

    @property
    def tangents(self) -> Float[jax.Array, "nodes 3"]:
        """Recover unit tangent directions from homogeneous derivatives."""
        velocities = self._velocities()
        return velocities / jnp.linalg.vector_norm(velocities, axis=-1, keepdims=True)

    @property
    def curvatures(self) -> Float[jax.Array, "nodes 3"]:
        """Recover curvature vectors from homogeneous endpoint derivatives."""
        f = self.homogeneous_f
        d1 = self.homogeneous_d1
        c0 = f[:, :, 0]
        c1 = c0 + d1[:, :, 0] / 3.0
        c3 = f[:, :, 1]
        c2 = c3 - d1[:, :, 1] / 3.0
        second_start = 6.0 * (c2 - 2.0 * c1 + c0)
        second_end = 6.0 * (c3 - 2.0 * c2 + c1)
        second = jnp.concatenate((second_start[:1], second_end), axis=0)

        values, derivatives = self._node_data()
        xyz, w = values[:, :3], values[:, 3, None]
        dxyz, dw = derivatives[:, :3], derivatives[:, 3, None]
        ddxyz, ddw = second[:, :3], second[:, 3, None]
        positions = xyz / w
        velocities = (dxyz - positions * dw) / w
        accelerations = (ddxyz - positions * ddw - 2.0 * velocities * dw) / w
        tangents = velocities / jnp.linalg.vector_norm(
            velocities, axis=-1, keepdims=True
        )
        normal_acceleration = (
            accelerations
            - jnp.sum(accelerations * tangents, axis=-1, keepdims=True) * tangents
        )
        return normal_acceleration / jnp.sum(velocities**2, axis=-1, keepdims=True)

    def interpolant(self, segment_index: int) -> Shaped[RationalBernstein, ""]:
        """Reconstruct one cubic rational Bernstein segment from Hermite data."""
        index = jnp.asarray(segment_index)
        if index.shape != ():
            raise ValueError("segment_index must be a scalar")
        index = index.astype(jnp.int32)
        index = eqx.error_if(
            index,
            (index < 0) | (index >= self.segment_count),
            "segment_index is outside the curve",
        )
        homogeneous = hermite_interpolate_1d(
            self.homogeneous_f[index], self.homogeneous_d1[index]
        ).c
        numerator = homogeneous[:3]
        denominator = jnp.broadcast_to(homogeneous[3], numerator.shape)
        return _from_homogeneous(numerator, denominator)

    def translated(self, offset: Float[jax.Array, "3"]) -> "PiecewiseRationalCurve3D":
        """Return a copy translated directly in homogeneous coordinates."""
        offset = jnp.asarray(offset, dtype=self.homogeneous_f.dtype)
        if offset.shape != (3,):
            raise ValueError(f"offset must have shape (3,), got {offset.shape}")
        offset = eqx.error_if(
            offset, jnp.any(~jnp.isfinite(offset)), "offset must be finite"
        )

        def translate(group):
            xyz = group[:, :3] + offset[None, :, None] * group[:, 3:4]
            return jnp.concatenate((xyz, group[:, 3:4]), axis=1)

        return type(self)._from_hermite(
            translate(self.homogeneous_f), translate(self.homogeneous_d1)
        )
