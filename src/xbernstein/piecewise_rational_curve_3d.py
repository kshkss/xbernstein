r"""Piecewise G² cubic rational Bernstein curves in three dimensions."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Shaped

from .rational_bernstein import RationalBernstein


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


class PiecewiseRationalCurve3D(eqx.Module):
    r"""Store a chain of 3D G² cubic rational Bernstein curve segments.

    ``positions``, ``tangents``, and ``curvatures`` all have shape
    ``(segment_count + 1, 3)``.  Tangents are normalized on construction;
    curvature vectors are projected onto their tangent-normal planes.  Thus
    neighboring segments share their endpoint position, oriented tangent, and
    curvature vector exactly.

    :meth:`interpolant` constructs one degree-three rational Bernstein curve.
    Its returned :class:`RationalBernstein` has three leading value axes and
    therefore evaluates to a point with shape ``(3,)``.
    """

    positions: Float[jax.Array, "nodes 3"]
    tangents: Float[jax.Array, "nodes 3"]
    curvatures: Float[jax.Array, "nodes 3"]

    def __init__(
        self,
        positions: Float[jax.Array, "nodes 3"],
        tangents: Float[jax.Array, "nodes 3"],
        curvatures: Float[jax.Array, "nodes 3"],
    ):
        groups = tuple(jnp.asarray(group) for group in (positions, tangents, curvatures))
        if any(group.ndim != 2 or group.shape[1] != 3 for group in groups):
            raise ValueError(
                "positions, tangents, and curvatures must each have shape (node_count, 3)"
            )
        if len({group.shape[0] for group in groups}) != 1:
            raise ValueError("positions, tangents, and curvatures must have equal node counts")
        if groups[0].shape[0] < 2:
            raise ValueError("at least two nodes are required")

        dtype = jnp.result_type(*groups, jnp.float32)
        positions, tangents, curvatures = (group.astype(dtype) for group in groups)
        finite = jnp.all(jnp.isfinite(positions)) & jnp.all(jnp.isfinite(tangents)) & jnp.all(
            jnp.isfinite(curvatures)
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
        curvatures = curvatures - jnp.sum(curvatures * tangents, axis=-1, keepdims=True) * tangents

        segment_lengths = jnp.linalg.vector_norm(jnp.diff(positions, axis=0), axis=-1)
        positions = eqx.error_if(
            positions,
            jnp.any(segment_lengths <= jnp.finfo(dtype).eps),
            "curve segments must have distinct endpoints",
        )
        self.positions = positions
        self.tangents = tangents
        self.curvatures = curvatures

    @property
    def segment_count(self) -> int:
        """Return the number of interpolable intervals."""
        return self.positions.shape[0] - 1

    @property
    def dtype(self) -> str:
        """Return the scalar dtype used by the node data."""
        return str(self.positions.dtype)

    def interpolant(self, segment_index: int) -> Shaped[RationalBernstein, ""]:
        r"""Return the G² cubic rational Bernstein interpolant for one interval.

        The endpoint weights are normalized to one.  The two interior control
        points lie on the supplied endpoint tangents; their offsets and the
        two positive interior weights are obtained from the endpoint curvature
        equations.  If these geometric constraints have no positive-weight
        cubic rational solution, an error is raised.
        """
        index = jnp.asarray(segment_index)
        if index.shape != ():
            raise ValueError("segment_index must be a scalar")
        index = index.astype(jnp.int32)
        index = eqx.error_if(
            index,
            (index < 0) | (index >= self.segment_count),
            "segment_index is outside the curve",
        )
        p0, p3 = self.positions[index], self.positions[index + 1]
        t0, t1 = self.tangents[index], self.tangents[index + 1]
        k0, k1 = self.curvatures[index], self.curvatures[index + 1]
        chord = p3 - p0
        tolerance = 256.0 * jnp.finfo(self.positions.dtype).eps

        # The all-straight case is represented by a degree-elevated line.
        straight = (jnp.linalg.vector_norm(k0) <= tolerance) & (
            jnp.linalg.vector_norm(k1) <= tolerance
        )
        chord_direction = chord / jnp.linalg.vector_norm(chord)
        straight_valid = (
            jnp.linalg.vector_norm(jnp.cross(t0, chord_direction)) <= tolerance
        ) & (jnp.linalg.vector_norm(jnp.cross(t1, chord_direction)) <= tolerance) & (
            jnp.dot(t0, chord_direction) > 0.0
        ) & (jnp.dot(t1, chord_direction) > 0.0)

        def make_straight(_):
            values = jnp.stack((p0, p0 + chord / 3.0, p0 + 2.0 * chord / 3.0, p3), axis=-1)
            return values, jnp.ones(4, dtype=self.positions.dtype), jnp.array(False)

        def make_curved(_):
            # P2 - P0 = chord - beta*t1 = mu0*k0 in t0's normal plane.
            start_matrix = jnp.stack((_orthogonal(t1, t0), k0), axis=-1)
            start_solution, start_residual = _solve_two_columns(
                start_matrix, _orthogonal(chord, t0)
            )
            beta, mu0 = start_solution

            # P1 - P3 = -chord + alpha*t0 = mu1*k1 in t1's normal plane.
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
            weights = jnp.stack((jnp.array(1.0, dtype=w1.dtype), w1, w2, jnp.array(1.0, dtype=w1.dtype)))
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

    def translated(
        self, offset: Float[jax.Array, "3"]
    ) -> "PiecewiseRationalCurve3D":
        """Return a copy translated by the three-dimensional ``offset``."""
        offset = jnp.asarray(offset, dtype=self.positions.dtype)
        if offset.shape != (3,):
            raise ValueError(f"offset must have shape (3,), got {offset.shape}")
        offset = eqx.error_if(offset, jnp.any(~jnp.isfinite(offset)), "offset must be finite")
        return type(self)(self.positions + offset, self.tangents, self.curvatures)
