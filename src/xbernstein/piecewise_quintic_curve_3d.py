r"""Piecewise C¹ quintic 3D curves parameterized by curvature and torsion.

Each node carries a position and a velocity (the shared C¹ Hermite jet used
by :class:`~xbernstein.c1_grid.P5C1Grid1D`). Each segment additionally
carries, at both of its ends, a curvature vector and a scalar torsion value.
Position and velocity determine four of a segment's six quintic Bezier
controls exactly as :class:`~xbernstein.c1_grid.P5C1Grid1D` already does; the
remaining two free controls (its ``interior_coefficients``) are solved here
so that the assembled segment reproduces the requested curvature vector and
torsion at each of its own endpoints.

Because those two free controls are independent per segment, only position
and velocity are guaranteed continuous across a shared node (matching
``P5C1Grid1D``'s own "C1" name): curvature and torsion may jump between two
segments that meet at the same point.
"""

import equinox as eqx
from beartype import beartype
import jax
import jax.numpy as jnp
from jaxtyping import Float, jaxtyped

from .c1_grid import P5C1Grid1D


def _orthogonal(
    vector: Float[jax.Array, "3"], tangent: Float[jax.Array, "3"]
) -> Float[jax.Array, "3"]:
    """Return the component of ``vector`` normal to the unit ``tangent``."""
    return vector - jnp.dot(vector, tangent) * tangent


def _solve_quintic_segment_interior(p0, v0, p1, v1, kappa0, tau0, kappa1, tau1):
    r"""Solve one segment's two free quintic Bezier controls.

    ``p0, v0, p1, v1`` fix the four boundary controls
    ``c0, c1, c4, c5`` exactly as :class:`~xbernstein.c1_grid.P5C1Grid1D`
    does. The two free controls ``c2, c3`` are chosen so that the resulting
    degree-5 Bezier segment has curvature vector ``kappa0``/``kappa1`` (classical,
    speed-independent ``normal_acceleration / |velocity|^2``) and geometric
    torsion ``tau0``/``tau1`` at ``t=0``/``t=1``.

    The acceleration at each end decomposes as
    ``a = kappa * |v|^2 + alpha * (v / |v|)``, where the tangential scalar
    ``alpha`` is not supplied directly: it is solved for from the torsion
    equations, since the binormal direction ``v x a`` does not depend on
    ``alpha`` (``v x (alpha * v/|v|) = 0``). This yields a well-posed 2x2
    linear system in the two unknown tangential accelerations.
    """
    dtype = jnp.result_type(p0, v0, p1, v1, kappa0, tau0, kappa1, tau1, jnp.float32)
    p0, v0, p1, v1, kappa0, kappa1 = (
        jnp.asarray(x, dtype=dtype) for x in (p0, v0, p1, v1, kappa0, kappa1)
    )
    tau0 = jnp.asarray(tau0, dtype=dtype)
    tau1 = jnp.asarray(tau1, dtype=dtype)
    tolerance = 256.0 * jnp.finfo(dtype).eps

    speed0 = jnp.linalg.vector_norm(v0)
    speed1 = jnp.linalg.vector_norm(v1)
    safe_speed0 = jnp.where(speed0 > tolerance, speed0, 1.0)
    safe_speed1 = jnp.where(speed1 > tolerance, speed1, 1.0)
    t0 = v0 / safe_speed0
    t1 = v1 / safe_speed1
    kappa0 = _orthogonal(kappa0, t0)
    kappa1 = _orthogonal(kappa1, t1)

    a0_normal = kappa0 * speed0**2
    a1_normal = kappa1 * speed1**2

    c0 = p0
    c1 = p0 + v0 / 5.0
    c4 = p1 - v1 / 5.0
    c5 = p1
    start = p0 + 2.0 * v0 / 5.0 + a0_normal / 20.0
    end = p1 - 2.0 * v1 / 5.0 + a1_normal / 20.0

    binormal0 = jnp.cross(v0, a0_normal)
    binormal1 = jnp.cross(v1, a1_normal)

    known0 = 60.0 * end - 180.0 * start + 180.0 * c1 - 60.0 * c0
    known1 = 60.0 * c5 - 180.0 * c4 + 180.0 * end - 60.0 * start

    matrix = jnp.array(
        [
            [-9.0 * jnp.dot(binormal0, t0), 3.0 * jnp.dot(binormal0, t1)],
            [-3.0 * jnp.dot(binormal1, t0), 9.0 * jnp.dot(binormal1, t1)],
        ]
    )
    rhs = jnp.array(
        [
            tau0 * jnp.dot(binormal0, binormal0) - jnp.dot(binormal0, known0),
            tau1 * jnp.dot(binormal1, binormal1) - jnp.dot(binormal1, known1),
        ]
    )

    valid = (
        (speed0 > tolerance)
        & (speed1 > tolerance)
        & (jnp.linalg.vector_norm(kappa0) > tolerance)
        & (jnp.linalg.vector_norm(kappa1) > tolerance)
        & (jnp.abs(jnp.linalg.det(matrix)) > tolerance)
    )
    safe_matrix = jnp.where(valid, matrix, jnp.eye(2, dtype=matrix.dtype))
    tangential = jnp.linalg.solve(safe_matrix, rhs)
    valid = valid & jnp.all(jnp.isfinite(tangential))

    c2 = start + (tangential[0] / 20.0) * t0
    c3 = end + (tangential[1] / 20.0) * t1
    c2 = eqx.error_if(
        c2,
        ~valid,
        "segment endpoint data (zero velocity, zero curvature, or degenerate "
        "curvature/torsion geometry) do not admit a quintic interior solve",
    )
    return c2, c3


@jaxtyped(typechecker=beartype)
class PiecewiseQuinticCurve3D(eqx.Module):
    r"""Store a chain of quintic segments built from curvature and torsion.

    The constructor takes one segment's worth of data: the position and
    velocity of both endpoints, and the curvature vector and torsion at both
    ends of the segment. :meth:`append` grows the curve by one point and one
    segment, returning a new instance rather than mutating this one.

    :meth:`to_p5c1grid1d` assembles the stored data into a
    :class:`~xbernstein.c1_grid.P5C1Grid1D` for evaluation; :meth:`__call__`
    does so and evaluates in one step.
    """

    positions: Float[jax.Array, "nodes 3"]
    velocities: Float[jax.Array, "nodes 3"]
    curvature_starts: Float[jax.Array, "segments 3"]
    curvature_ends: Float[jax.Array, "segments 3"]
    torsion_starts: Float[jax.Array, "segments"]
    torsion_ends: Float[jax.Array, "segments"]

    def __init__(
        self,
        p0: Float[jax.Array, "3"],
        v0: Float[jax.Array, "3"],
        p1: Float[jax.Array, "3"],
        v1: Float[jax.Array, "3"],
        curvature0: Float[jax.Array, "3"],
        torsion0: Float[jax.Array, ""],
        curvature1: Float[jax.Array, "3"],
        torsion1: Float[jax.Array, ""],
    ):
        self._build(
            jnp.stack((p0, p1)),
            jnp.stack((v0, v1)),
            jnp.asarray(curvature0)[None],
            jnp.asarray(curvature1)[None],
            jnp.asarray(torsion0)[None],
            jnp.asarray(torsion1)[None],
        )

    def _build(
        self,
        positions,
        velocities,
        curvature_starts,
        curvature_ends,
        torsion_starts,
        torsion_ends,
    ):
        positions = jnp.asarray(positions)
        velocities = jnp.asarray(velocities)
        curvature_starts = jnp.asarray(curvature_starts)
        curvature_ends = jnp.asarray(curvature_ends)
        torsion_starts = jnp.asarray(torsion_starts)
        torsion_ends = jnp.asarray(torsion_ends)

        if positions.ndim != 2 or positions.shape[-1] != 3 or positions.shape[0] < 2:
            raise ValueError(
                "positions must have shape (node_count, 3) with node_count >= 2"
            )
        node_count = positions.shape[0]
        segment_count = node_count - 1
        segment_shape = (segment_count, 3)

        if velocities.shape != positions.shape:
            raise ValueError(f"velocities must have shape {positions.shape}")
        if curvature_starts.shape != segment_shape:
            raise ValueError(f"curvature_starts must have shape {segment_shape}")
        if curvature_ends.shape != segment_shape:
            raise ValueError(f"curvature_ends must have shape {segment_shape}")
        if torsion_starts.shape != (segment_count,):
            raise ValueError(f"torsion_starts must have shape ({segment_count},)")
        if torsion_ends.shape != (segment_count,):
            raise ValueError(f"torsion_ends must have shape ({segment_count},)")

        dtype = jnp.result_type(
            positions,
            velocities,
            curvature_starts,
            curvature_ends,
            torsion_starts,
            torsion_ends,
            jnp.float32,
        )
        positions = positions.astype(dtype)
        velocities = velocities.astype(dtype)
        curvature_starts = curvature_starts.astype(dtype)
        curvature_ends = curvature_ends.astype(dtype)
        torsion_starts = torsion_starts.astype(dtype)
        torsion_ends = torsion_ends.astype(dtype)

        finite = (
            jnp.all(jnp.isfinite(positions))
            & jnp.all(jnp.isfinite(velocities))
            & jnp.all(jnp.isfinite(curvature_starts))
            & jnp.all(jnp.isfinite(curvature_ends))
            & jnp.all(jnp.isfinite(torsion_starts))
            & jnp.all(jnp.isfinite(torsion_ends))
        )
        positions = eqx.error_if(
            positions, ~finite, "curve data must contain only finite values"
        )
        velocity_norms = jnp.linalg.vector_norm(velocities, axis=-1)
        velocities = eqx.error_if(
            velocities,
            jnp.any(velocity_norms <= jnp.finfo(dtype).eps),
            "curve velocities must be non-zero",
        )

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "curvature_starts", curvature_starts)
        object.__setattr__(self, "curvature_ends", curvature_ends)
        object.__setattr__(self, "torsion_starts", torsion_starts)
        object.__setattr__(self, "torsion_ends", torsion_ends)

    @classmethod
    def _from_arrays(
        cls,
        positions,
        velocities,
        curvature_starts,
        curvature_ends,
        torsion_starts,
        torsion_ends,
    ) -> "PiecewiseQuinticCurve3D":
        result = object.__new__(cls)
        result._build(
            positions,
            velocities,
            curvature_starts,
            curvature_ends,
            torsion_starts,
            torsion_ends,
        )
        return result

    def append(
        self,
        position: Float[jax.Array, "3"],
        velocity: Float[jax.Array, "3"],
        curvature_start: Float[jax.Array, "3"],
        torsion_start: Float[jax.Array, ""],
        curvature_end: Float[jax.Array, "3"],
        torsion_end: Float[jax.Array, ""],
    ) -> "PiecewiseQuinticCurve3D":
        """Return a new curve with one more point and one more segment appended."""
        return type(self)._from_arrays(
            jnp.concatenate((self.positions, jnp.asarray(position)[None])),
            jnp.concatenate((self.velocities, jnp.asarray(velocity)[None])),
            jnp.concatenate((self.curvature_starts, jnp.asarray(curvature_start)[None])),
            jnp.concatenate((self.curvature_ends, jnp.asarray(curvature_end)[None])),
            jnp.concatenate((self.torsion_starts, jnp.asarray(torsion_start)[None])),
            jnp.concatenate((self.torsion_ends, jnp.asarray(torsion_end)[None])),
        )

    @property
    def node_count(self) -> int:
        """Return the number of stored points."""
        return self.positions.shape[0]

    @property
    def segment_count(self) -> int:
        """Return the number of segments."""
        return self.node_count - 1

    def to_p5c1grid1d(self) -> P5C1Grid1D:
        """Assemble the stored data into a :class:`~xbernstein.c1_grid.P5C1Grid1D`."""
        node_count = self.node_count
        segment_count = self.segment_count
        nodes = jnp.arange(node_count, dtype=self.positions.dtype)
        c2, c3 = jax.vmap(_solve_quintic_segment_interior)(
            self.positions[:-1],
            self.velocities[:-1],
            self.positions[1:],
            self.velocities[1:],
            self.curvature_starts,
            self.torsion_starts,
            self.curvature_ends,
            self.torsion_ends,
        )
        dof_size = 5 * segment_count + 1
        segment_index = jnp.arange(segment_count)
        interior_coefficients = jnp.zeros((3, dof_size), dtype=self.positions.dtype)
        interior_coefficients = interior_coefficients.at[:, segment_index * 5 + 2].set(c2.T)
        interior_coefficients = interior_coefficients.at[:, segment_index * 5 + 3].set(c3.T)
        return P5C1Grid1D(
            nodes, self.positions.T, self.velocities.T, interior_coefficients
        )

    def __call__(self, point):
        """Evaluate the assembled curve at ``point``."""
        grid = self.to_p5c1grid1d()
        point = jnp.asarray(point, dtype=self.positions.dtype)
        flat = point.reshape(-1)
        values = jax.vmap(lambda t: grid(t[None]))(flat)
        return values.reshape(point.shape + (3,))

    def _frenet_frame(self, grid, t):
        """Return the unit tangent, principal normal, and binormal at scalar ``t``."""
        t_arr = t[None]
        cell = grid.cell_index(t_arr)
        local = grid._local_coordinates(t_arr, cell)
        poly = grid.cell_interpolant(cell)
        velocity = poly.deriv(1)(*local)
        acceleration = poly.deriv(2)(*local)

        tangent = velocity / jnp.linalg.vector_norm(velocity)
        normal_component = acceleration - jnp.dot(acceleration, tangent) * tangent
        normal_norm = jnp.linalg.vector_norm(normal_component)
        tolerance = 256.0 * jnp.finfo(self.positions.dtype).eps
        normal_component = eqx.error_if(
            normal_component,
            normal_norm <= tolerance,
            "the principal normal and binormal are undefined where the curve "
            "has zero curvature",
        )
        normal = normal_component / normal_norm
        binormal = jnp.cross(tangent, normal)
        return tangent, normal, binormal

    def _evaluate_frenet(self, point, index):
        grid = self.to_p5c1grid1d()
        point = jnp.asarray(point, dtype=self.positions.dtype)
        flat = point.reshape(-1)
        vectors = jax.vmap(lambda t: self._frenet_frame(grid, t)[index])(flat)
        return vectors.reshape(point.shape + (3,))

    def tangent(self, point):
        """Return the unit tangent vector at ``point``."""
        return self._evaluate_frenet(point, 0)

    def normal(self, point):
        """Return the unit principal normal vector at ``point``."""
        return self._evaluate_frenet(point, 1)

    def binormal(self, point):
        """Return the unit binormal vector at ``point``."""
        return self._evaluate_frenet(point, 2)
