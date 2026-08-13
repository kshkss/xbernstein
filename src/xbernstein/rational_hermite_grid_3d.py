r"""Tricubic rational Hermite interpolation on a rectilinear 3D grid."""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .hermite import _multi_indices
from .rational_hermite import rational_hermite_interpolate_3d
from .rational_tensor_bernstein import RationalBernstein3D


class GridSegment3D(NamedTuple):
    r"""Store a fixed-capacity decomposition of a segment into grid cells.

    ``cell_indices`` has shape ``(M, 3)`` and ``local_endpoints`` has shape
    ``(M, 2, 3)``. Only rows selected by ``valid_mask`` are meaningful.
    Valid rows are ordered from the requested segment start to its end.
    """

    cell_indices: jax.Array
    local_endpoints: jax.Array
    valid_mask: jax.Array


class RationalHermiteGrid3D(eqx.Module):
    r"""Store scalar vertex jets on a nonuniform rectilinear 3D grid.

    ``x``, ``y``, and ``z`` are strictly increasing coordinate arrays. ``f``
    has shape ``(*batch, nx, ny, nz)``. Derivative groups ``d1`` through
    ``d6`` use the same ordering as :func:`rational_hermite_interpolate_3d`
    and have derivative-kind counts ``3, 6, 7, 6, 3, 1``. Leading batch axes
    must be identical for every group and are preserved by evaluation.

    Query points have shape ``(3,)`` and are clipped componentwise to the
    closed grid domain. Interior grid planes belong to the cell on their
    positive side; the upper domain endpoint belongs to the final cell.
    Physical derivatives are scaled by the selected cell widths before the
    local :math:`[0,1]^3` patch is constructed.
    """

    x: jax.Array
    y: jax.Array
    z: jax.Array
    f: jax.Array
    d1: jax.Array
    d2: jax.Array
    d3: jax.Array
    d4: jax.Array
    d5: jax.Array
    d6: jax.Array

    def __init__(self, x, y, z, f, d1, d2, d3, d4, d5, d6):
        coordinates = tuple(jnp.asarray(axis) for axis in (x, y, z))
        for name, axis in zip(("x", "y", "z"), coordinates):
            if axis.ndim != 1 or axis.shape[0] < 2:
                raise ValueError(f"{name} must be a one-dimensional array of length >= 2")
            axis = eqx.error_if(
                axis,
                jnp.any(~jnp.isfinite(axis)) | jnp.any(jnp.diff(axis) <= 0.0),
                f"{name} must contain finite, strictly increasing coordinates",
            )
            object.__setattr__(self, name, axis)

        groups = tuple(jnp.asarray(group) for group in (f, d1, d2, d3, d4, d5, d6))
        grid_shape = tuple(axis.shape[0] for axis in coordinates)
        if groups[0].ndim < 3 or groups[0].shape[-3:] != grid_shape:
            raise ValueError(f"f must end with grid shape {grid_shape}")
        batch_shape = groups[0].shape[:-3]
        counts = (3, 6, 7, 6, 3, 1)
        for total, (group, count) in enumerate(zip(groups[1:], counts), start=1):
            expected = batch_shape + grid_shape + (() if count == 1 else (count,))
            if group.shape != expected:
                raise ValueError(f"d{total} must have shape {expected}")

        dtype = jnp.result_type(*groups, *coordinates, jnp.float32)
        for name, group in zip(("f", "d1", "d2", "d3", "d4", "d5", "d6"), groups):
            group = group.astype(dtype)
            group = eqx.error_if(
                group,
                jnp.any(~jnp.isfinite(group)),
                f"{name} must contain only finite values",
            )
            object.__setattr__(self, name, group)

        for name in ("x", "y", "z"):
            object.__setattr__(self, name, getattr(self, name).astype(dtype))

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the leading field batch shape."""
        return self.f.shape[:-3]

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Return the number of vertices along each grid axis."""
        return (self.x.shape[0], self.y.shape[0], self.z.shape[0])

    @property
    def segment_capacity(self) -> int:
        """Return the maximum number of cells crossed by one segment."""
        return sum(size - 1 for size in self.grid_shape) - 2

    def _point(self, point) -> jax.Array:
        point = jnp.asarray(point, dtype=self.f.dtype)
        if point.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {point.shape}")
        lower = jnp.asarray([self.x[0], self.y[0], self.z[0]])
        upper = jnp.asarray([self.x[-1], self.y[-1], self.z[-1]])
        return jnp.clip(point, lower, upper)

    def cell_index(self, point) -> jax.Array:
        """Return the index of the cell containing the clipped physical point."""
        point = self._point(point)
        indices = [
            jnp.clip(jnp.searchsorted(axis, value, side="right") - 1, 0, axis.size - 2)
            for axis, value in zip((self.x, self.y, self.z), point)
        ]
        return jnp.stack(indices).astype(jnp.int32)

    def _local_coordinates(self, point: jax.Array, cell_index: jax.Array) -> jax.Array:
        axes = (self.x, self.y, self.z)
        starts = jnp.stack([axis[index] for axis, index in zip(axes, cell_index)])
        ends = jnp.stack([axis[index + 1] for axis, index in zip(axes, cell_index)])
        return jnp.clip((point - starts) / (ends - starts), 0.0, 1.0)

    def _cell_vertices(self, group: jax.Array, cell_index: jax.Array) -> jax.Array:
        result = group
        batch_dimensions = len(self.shape)
        for axis_offset in range(3):
            indices = cell_index[axis_offset] + jnp.arange(2)
            result = jnp.take(result, indices, axis=batch_dimensions + axis_offset)
        return result

    def cell_interpolant(self, cell_index) -> RationalBernstein3D:
        """Return the tricubic rational Bernstein interpolant for one cell."""
        cell_index = jnp.asarray(cell_index)
        if cell_index.shape != (3,):
            raise ValueError(
                f"cell_index must have shape (3,), got {cell_index.shape}"
            )
        maximum = jnp.asarray(self.grid_shape, dtype=jnp.int32) - 2
        cell_index = cell_index.astype(jnp.int32)
        cell_index = eqx.error_if(
            cell_index,
            jnp.any(cell_index < 0) | jnp.any(cell_index > maximum),
            "cell_index is outside the grid",
        )
        axes = (self.x, self.y, self.z)
        widths = jnp.stack(
            [axis[index + 1] - axis[index] for axis, index in zip(axes, cell_index)]
        )

        source_groups = (self.f, self.d1, self.d2, self.d3, self.d4, self.d5, self.d6)
        local_groups = [self._cell_vertices(self.f, cell_index)]
        for total, source in enumerate(source_groups[1:], start=1):
            group = self._cell_vertices(source, cell_index)
            alphas = _multi_indices(3, 3, total)
            scales = jnp.stack(
                [jnp.prod(widths ** jnp.asarray(alpha)) for alpha in alphas]
            )
            if len(alphas) == 1:
                group = group * scales[0]
            else:
                scale_shape = (1,) * (group.ndim - 1) + (len(alphas),)
                group = group * scales.reshape(scale_shape)
            local_groups.append(group)
        return rational_hermite_interpolate_3d(*local_groups)

    def __call__(self, point):
        """Evaluate the clipped point with its cell's rational interpolant."""
        point = self._point(point)
        cell_index = self.cell_index(point)
        local = self._local_coordinates(point, cell_index)
        return self.cell_interpolant(cell_index)(local[0], local[1], local[2])

    def split_segment(self, start, end) -> GridSegment3D:
        r"""Split a clipped segment into fixed-capacity cell-local pieces.

        The capacity is ``(nx - 1) + (ny - 1) + (nz - 1) - 2``. Invalid
        padding rows must be ignored using :attr:`GridSegment3D.valid_mask`.
        A zero-length segment is represented by one valid zero-length piece.
        """
        start = self._point(start)
        end = self._point(end)
        delta = end - start

        candidates = [jnp.asarray([0.0, 1.0], dtype=self.f.dtype)]
        for axis_index, axis in enumerate((self.x, self.y, self.z)):
            denominator = delta[axis_index]
            parameters = jnp.where(
                denominator != 0.0,
                (axis[1:-1] - start[axis_index]) / denominator,
                jnp.inf,
            )
            parameters = jnp.where(
                (parameters > 0.0) & (parameters < 1.0), parameters, jnp.inf
            )
            candidates.append(parameters)
        parameters = jnp.sort(jnp.concatenate(candidates))
        interval_starts = parameters[:-1]
        interval_ends = parameters[1:]
        tolerance = 16.0 * jnp.finfo(self.f.dtype).eps
        interval_valid = (
            jnp.isfinite(interval_starts)
            & jnp.isfinite(interval_ends)
            & (interval_ends - interval_starts > tolerance)
        )

        capacity = self.segment_capacity
        cell_indices = jnp.zeros((capacity, 3), dtype=jnp.int32)
        local_endpoints = jnp.zeros((capacity, 2, 3), dtype=self.f.dtype)
        valid_mask = jnp.zeros((capacity,), dtype=bool)

        def add_interval(candidate_index, state):
            output_index, cells, endpoints, mask = state
            valid = interval_valid[candidate_index]
            parameter_pair = jnp.stack(
                (interval_starts[candidate_index], interval_ends[candidate_index])
            )
            physical = start + parameter_pair[:, None] * delta
            midpoint = jnp.mean(physical, axis=0)
            cell = self.cell_index(midpoint)
            local = jax.vmap(lambda point: self._local_coordinates(point, cell))(physical)

            def write(current):
                index, current_cells, current_endpoints, current_mask = current
                current_cells = current_cells.at[index].set(cell)
                current_endpoints = current_endpoints.at[index].set(local)
                current_mask = current_mask.at[index].set(True)
                return index + 1, current_cells, current_endpoints, current_mask

            return jax.lax.cond(
                valid & (output_index < capacity),
                write,
                lambda current: current,
                (output_index, cells, endpoints, mask),
            )

        _, cell_indices, local_endpoints, valid_mask = jax.lax.fori_loop(
            0,
            interval_starts.shape[0],
            add_interval,
            (jnp.asarray(0, dtype=jnp.int32), cell_indices, local_endpoints, valid_mask),
        )

        no_piece = ~jnp.any(valid_mask)
        cell = self.cell_index(start)
        local = self._local_coordinates(start, cell)
        cell_indices = cell_indices.at[0].set(jnp.where(no_piece, cell, cell_indices[0]))
        local_endpoints = local_endpoints.at[0].set(
            jnp.where(no_piece, jnp.stack((local, local)), local_endpoints[0])
        )
        valid_mask = valid_mask.at[0].set(valid_mask[0] | no_piece)
        return GridSegment3D(cell_indices, local_endpoints, valid_mask)
