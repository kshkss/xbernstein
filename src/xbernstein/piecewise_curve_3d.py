r"""Piecewise C1 cubic Hermite curves in three dimensions."""

import equinox as eqx
from beartype import beartype
import jax
import jax.numpy as jnp
from jaxtyping import Float, Shaped, jaxtyped

from .bernstein import Bernstein


@jaxtyped(typechecker=beartype)
class PiecewiseCurve3D(eqx.Module):
    r"""Store a piecewise cubic Hermite curve in three dimensions.

    ``positions`` and ``tangents`` have shape ``(node_count, 3)``. Tangents
    are used as derivatives with respect to each segment's local parameter;
    they are not normalized. Each segment is a vector-valued cubic Bernstein
    curve, and neighboring segments are C1 continuous.
    """

    positions: Float[jax.Array, "nodes 3"]
    tangents: Float[jax.Array, "nodes 3"]

    def __init__(self, positions, tangents):
        positions = jnp.asarray(positions)
        tangents = jnp.asarray(tangents)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (node_count, 3)")
        if tangents.ndim != 2 or tangents.shape[1] != 3:
            raise ValueError("tangents must have shape (node_count, 3)")
        if positions.shape[0] != tangents.shape[0]:
            raise ValueError("positions and tangents must have equal node counts")
        if positions.shape[0] < 2:
            raise ValueError("at least two nodes are required")

        dtype = jnp.result_type(positions, tangents, jnp.float32)
        positions = positions.astype(dtype)
        tangents = tangents.astype(dtype)
        positions = eqx.error_if(
            positions,
            jnp.any(~jnp.isfinite(positions)),
            "positions must contain only finite values",
        )
        tangents = eqx.error_if(
            tangents,
            jnp.any(~jnp.isfinite(tangents)),
            "tangents must contain only finite values",
        )
        self.positions = positions
        self.tangents = tangents

    @property
    def segment_count(self) -> int:
        """Return the number of interpolable curve segments."""
        return self.positions.shape[0] - 1

    @property
    def dtype(self) -> str:
        """Return the scalar dtype used by the curve."""
        return str(self.positions.dtype)

    def interpolant(self, segment_index: int) -> Shaped[Bernstein, ""]:
        """Return the vector-valued cubic Bernstein curve for one segment."""
        index = jnp.asarray(segment_index)
        if index.shape != ():
            raise ValueError("segment_index must be a scalar")
        index = index.astype(jnp.int32)
        index = eqx.error_if(
            index,
            (index < 0) | (index >= self.segment_count),
            "segment_index is outside the curve",
        )
        p0 = self.positions[index]
        p1 = self.positions[index + 1]
        v0 = self.tangents[index]
        v1 = self.tangents[index + 1]
        coefficients = jnp.stack((p0, p0 + v0 / 3.0, p1 - v1 / 3.0, p1), axis=-1)
        return Bernstein(coefficients)

    def __call__(self, parameter):
        r"""Evaluate the curve at global parameter ``t``.

        The domain is ``[0, segment_count]``. Every unit interval selects one
        segment, and values outside the domain are clipped to its endpoints.
        Array parameters return values with shape ``parameter.shape + (3,)``.
        """
        parameter = jnp.asarray(parameter, dtype=self.positions.dtype)
        parameter = eqx.error_if(
            parameter,
            jnp.any(~jnp.isfinite(parameter)),
            "parameter must contain only finite values",
        )
        parameter = jnp.clip(parameter, 0.0, float(self.segment_count))
        flat = parameter.reshape(-1)
        indices = jnp.minimum(
            jnp.floor(flat).astype(jnp.int32), self.segment_count - 1
        )
        local = flat - indices

        def evaluate(index, value):
            return self.interpolant(index)(value)

        values = jax.vmap(evaluate)(indices, local)
        values = values.reshape(parameter.shape + (3,))
        return values
