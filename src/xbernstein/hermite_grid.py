r"""Cubic and quintic Hermite interpolation on rectilinear grids."""

import itertools
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Shaped

from .bernstein import Bernstein
from .bernstein_2d import Bernstein2D
from .bernstein_3d import Bernstein3D
from .bernstein_4d import Bernstein4D
from .hermite import (
    _multi_indices,
    hermite_interpolate_1d,
    hermite_interpolate_2d,
    hermite_interpolate_3d,
    hermite_interpolate_4d,
    quintic_hermite_interpolate_1d,
    quintic_hermite_interpolate_2d,
    quintic_hermite_interpolate_3d,
    quintic_hermite_interpolate_4d,
)


class GridSegment(NamedTuple):
    """Fixed-capacity decomposition of a segment into grid-cell pieces."""

    cell_indices: jax.Array
    local_endpoints: jax.Array
    valid_mask: jax.Array


class _HermiteGrid(eqx.Module):
    """Shared implementation for the public dimension-specific grid classes."""

    axes: tuple
    coefficients: jax.Array
    dimension: int = eqx.field(static=True)
    jet_size: int = eqx.field(static=True)

    def _initialize(self, axes, groups, dimension, jet_size=2):
        coordinates = [jnp.asarray(axis) for axis in axes]
        names = "xyzw"[:dimension]
        for axis_index, (name, axis) in enumerate(zip(names, coordinates)):
            if axis.ndim != 1 or axis.shape[0] < 2:
                raise ValueError(
                    f"{name} must be a one-dimensional array of length >= 2"
                )
            axis = eqx.error_if(
                axis,
                jnp.any(~jnp.isfinite(axis)) | jnp.any(jnp.diff(axis) <= 0.0),
                f"{name} must contain finite, strictly increasing coordinates",
            )
            coordinates[axis_index] = axis

        grid_shape = tuple(axis.shape[0] for axis in coordinates)
        values = tuple(jnp.asarray(group) for group in groups)
        expected_group_count = (jet_size - 1) * dimension + 1
        if len(values) != expected_group_count:
            raise TypeError(
                f"{dimension}D order-{jet_size - 1} Hermite grid requires "
                f"{expected_group_count} derivative groups"
            )
        if values[0].ndim < dimension or values[0].shape[-dimension:] != grid_shape:
            raise ValueError(f"f must end with grid shape {grid_shape}")
        batch_shape = values[0].shape[:-dimension]
        expected_counts = tuple(
            len(_multi_indices(dimension, jet_size, total))
            for total in range(1, (jet_size - 1) * dimension + 1)
        )
        for total, (group, count) in enumerate(
            zip(values[1:], expected_counts), start=1
        ):
            expected = batch_shape + grid_shape + (() if count == 1 else (count,))
            if group.shape != expected:
                raise ValueError(f"d{total} must have shape {expected}")

        dtype = jnp.result_type(*values, *coordinates, jnp.float32)
        converted = []
        derivative_names = ("f",) + tuple(
            f"d{i}" for i in range(1, expected_group_count)
        )
        for name, group in zip(derivative_names, values):
            group = group.astype(dtype)
            group = eqx.error_if(
                group,
                jnp.any(~jnp.isfinite(group)),
                f"{name} must contain only finite values",
            )
            converted.append(group)
        coordinates = tuple(axis.astype(dtype) for axis in coordinates)
        coefficients = self._build_coefficients(
            coordinates,
            tuple(converted),
            dimension,
            jet_size,
            batch_shape,
            grid_shape,
        )
        object.__setattr__(self, "axes", coordinates)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "jet_size", jet_size)

    @staticmethod
    def _build_coefficients(
        coordinates, groups, dimension, jet_size, batch_shape, grid_shape
    ):
        """Convert all physical vertex jets to cell-local Bernstein controls."""
        batch_dimensions = len(batch_shape)
        cell_shape = tuple(size - 1 for size in grid_shape)
        vertices = tuple(itertools.product((0, 1), repeat=dimension))

        def gather_cell_vertices(group):
            tail = group.shape[batch_dimensions + dimension :]
            cell_vertices = []
            for vertex in vertices:
                slices = (slice(None),) * batch_dimensions + tuple(
                    slice(offset, offset + size)
                    for offset, size in zip(vertex, cell_shape)
                )
                cell_vertices.append(group[slices])
            stacked = jnp.stack(
                cell_vertices,
                axis=batch_dimensions + dimension,
            )
            return stacked.reshape(batch_shape + cell_shape + (2,) * dimension + tail)

        width_fields = []
        for axis_index, axis in enumerate(coordinates):
            width_shape = [1] * dimension
            width_shape[axis_index] = cell_shape[axis_index]
            width_fields.append(jnp.diff(axis).reshape(width_shape))

        local_groups = [gather_cell_vertices(groups[0])]
        for total, source in enumerate(groups[1:], start=1):
            group = gather_cell_vertices(source)
            alphas = _multi_indices(dimension, jet_size, total)
            scales = [
                jnp.prod(
                    jnp.stack(
                        [
                            jnp.broadcast_to(width, cell_shape) ** order
                            for width, order in zip(width_fields, alpha)
                        ]
                    ),
                    axis=0,
                )
                for alpha in alphas
            ]
            if len(alphas) == 1:
                scale = scales[0].reshape(
                    (1,) * batch_dimensions + cell_shape + (1,) * dimension
                )
            else:
                scale = jnp.stack(scales, axis=-1).reshape(
                    (1,) * batch_dimensions
                    + cell_shape
                    + (1,) * dimension
                    + (len(alphas),)
                )
            local_groups.append(group * scale)

        cubic_interpolators = (
            hermite_interpolate_1d,
            hermite_interpolate_2d,
            hermite_interpolate_3d,
            hermite_interpolate_4d,
        )
        quintic_interpolators = (
            quintic_hermite_interpolate_1d,
            quintic_hermite_interpolate_2d,
            quintic_hermite_interpolate_3d,
            quintic_hermite_interpolate_4d,
        )
        interpolators = cubic_interpolators if jet_size == 2 else quintic_interpolators
        return interpolators[dimension - 1](*local_groups).c

    @property
    def f(self):
        """Reconstruct scalar values at the grid vertices."""
        return self._reconstructed_group(0)

    @property
    def x(self):
        return self.axes[0]

    @property
    def y(self):
        return self.axes[1]

    @property
    def z(self):
        return self.axes[2]

    @property
    def w(self):
        return self.axes[3]

    @property
    def d1(self):
        """Reconstruct the total-order-one derivative group."""
        return self._reconstructed_group(1)

    @property
    def d2(self):
        """Reconstruct the total-order-two derivative group."""
        return self._reconstructed_group(2)

    @property
    def d3(self):
        """Reconstruct the total-order-three derivative group."""
        return self._reconstructed_group(3)

    @property
    def d4(self):
        """Reconstruct the total-order-four derivative group."""
        return self._reconstructed_group(4)

    @property
    def d5(self):
        """Reconstruct the total-order-five derivative group."""
        return self._reconstructed_group(5)

    @property
    def d6(self):
        """Reconstruct the total-order-six derivative group."""
        return self._reconstructed_group(6)

    @property
    def d7(self):
        """Reconstruct the total-order-seven derivative group."""
        return self._reconstructed_group(7)

    @property
    def d8(self):
        """Reconstruct the total-order-eight derivative group."""
        return self._reconstructed_group(8)

    @property
    def shape(self):
        return self.coefficients.shape[: -2 * self.dimension]

    @property
    def dtype(self):
        """Return the scalar dtype used by the Bernstein coefficients."""
        return str(self.coefficients.dtype)

    @property
    def grid_shape(self):
        return tuple(axis.shape[0] for axis in self.axes)

    @property
    def segment_capacity(self):
        return sum(size - 1 for size in self.grid_shape) - self.dimension + 1

    def _point(self, point):
        point = jnp.asarray(point, dtype=self.coefficients.dtype)
        if point.shape != (self.dimension,):
            raise ValueError(
                f"point must have shape ({self.dimension},), got {point.shape}"
            )
        lower = jnp.stack([axis[0] for axis in self.axes])
        upper = jnp.stack([axis[-1] for axis in self.axes])
        return jnp.clip(point, lower, upper)

    def cell_index(self, point):
        point = self._point(point)
        indices = [
            jnp.clip(jnp.searchsorted(axis, value, side="right") - 1, 0, axis.size - 2)
            for axis, value in zip(self.axes, point)
        ]
        return jnp.stack(indices).astype(jnp.int32)

    def _local_coordinates(self, point, cell_index):
        starts = jnp.stack([axis[index] for axis, index in zip(self.axes, cell_index)])
        ends = jnp.stack(
            [axis[index + 1] for axis, index in zip(self.axes, cell_index)]
        )
        return jnp.clip((point - starts) / (ends - starts), 0.0, 1.0)

    def _cell_coefficients(self, cell_index):
        result = self.coefficients
        batch_dimensions = len(self.shape)
        for offset in range(self.dimension):
            result = jnp.take(
                result,
                cell_index[offset],
                axis=batch_dimensions,
            )
        return result

    def _polynomial_vertex_derivative(self, coefficients, alpha, vertex):
        """Evaluate one local polynomial derivative at a cell vertex."""
        result = coefficients
        degree = 2 * self.jet_size - 1
        for axis, order in enumerate(alpha):
            current_degree = degree
            for _ in range(order):
                result = current_degree * jnp.diff(
                    result,
                    axis=result.ndim - self.dimension + axis,
                )
                current_degree -= 1
        for axis in range(self.dimension - 1, -1, -1):
            endpoint = jnp.where(vertex[axis] == 0, 0, result.shape[-1] - 1)
            result = jnp.take(result, endpoint, axis=-1)
        return result

    def _reconstructed_group(self, total):
        """Reconstruct one physical derivative group from cell controls."""
        alphas = _multi_indices(self.dimension, self.jet_size, total)
        grid_shape = jnp.asarray(self.grid_shape, dtype=jnp.int32)
        grid_indices = jnp.stack(
            jnp.meshgrid(
                *(jnp.arange(size, dtype=jnp.int32) for size in self.grid_shape),
                indexing="ij",
            ),
            axis=-1,
        ).reshape((-1, self.dimension))

        def reconstruct(grid_index):
            cell_index = jnp.minimum(grid_index, grid_shape - 2)
            vertex = (grid_index == grid_shape - 1).astype(jnp.int32)
            coefficients = self._cell_coefficients(cell_index)
            widths = jnp.stack(
                [
                    axis[index + 1] - axis[index]
                    for axis, index in zip(self.axes, cell_index)
                ]
            )
            values = [
                self._polynomial_vertex_derivative(coefficients, alpha, vertex)
                / jnp.prod(widths ** jnp.asarray(alpha))
                for alpha in alphas
            ]
            return values[0] if len(values) == 1 else jnp.stack(values, axis=-1)

        reconstructed = jax.vmap(reconstruct)(grid_indices)
        batch_dimensions = len(self.shape)
        reconstructed = jnp.moveaxis(reconstructed, 0, batch_dimensions)
        tail = () if len(alphas) == 1 else (len(alphas),)
        return reconstructed.reshape(self.shape + self.grid_shape + tail)

    def cell_interpolant(self, cell_index) -> Shaped:
        cell_index = jnp.asarray(cell_index)
        if cell_index.shape != (self.dimension,):
            raise ValueError(
                f"cell_index must have shape ({self.dimension},), got {cell_index.shape}"
            )
        cell_index = cell_index.astype(jnp.int32)
        maximum = jnp.asarray(self.grid_shape, dtype=jnp.int32) - 2
        cell_index = eqx.error_if(
            cell_index,
            jnp.any(cell_index < 0) | jnp.any(cell_index > maximum),
            "cell_index is outside the grid",
        )
        polynomial_types = (Bernstein, Bernstein2D, Bernstein3D, Bernstein4D)
        return polynomial_types[self.dimension - 1](self._cell_coefficients(cell_index))

    def __call__(self, point):
        point = self._point(point)
        cell = self.cell_index(point)
        local = self._local_coordinates(point, cell)
        return self.cell_interpolant(cell)(*tuple(local))

    def _split_segment(self, start, end):
        start = self._point(start)
        end = self._point(end)
        delta = end - start
        candidates = [jnp.asarray([0.0, 1.0], dtype=self.coefficients.dtype)]
        for axis_index, axis in enumerate(self.axes):
            denominator = delta[axis_index]
            parameters = jnp.where(
                denominator != 0.0,
                (axis[1:-1] - start[axis_index]) / denominator,
                jnp.inf,
            )
            candidates.append(
                jnp.where((parameters > 0.0) & (parameters < 1.0), parameters, jnp.inf)
            )
        parameters = jnp.sort(jnp.concatenate(candidates))
        interval_starts, interval_ends = parameters[:-1], parameters[1:]
        tolerance = 16.0 * jnp.finfo(self.coefficients.dtype).eps
        valid = (
            jnp.isfinite(interval_starts)
            & jnp.isfinite(interval_ends)
            & (interval_ends - interval_starts > tolerance)
        )
        capacity = self.segment_capacity
        cells = jnp.zeros((capacity, self.dimension), dtype=jnp.int32)
        endpoints = jnp.zeros(
            (capacity, 2, self.dimension), dtype=self.coefficients.dtype
        )
        mask = jnp.zeros((capacity,), dtype=bool)

        def add(i, state):
            out, cells, endpoints, mask = state
            pair = jnp.stack((interval_starts[i], interval_ends[i]))
            physical = start + pair[:, None] * delta
            cell = self.cell_index(jnp.mean(physical, axis=0))
            local = jax.vmap(lambda p: self._local_coordinates(p, cell))(physical)

            def write(s):
                out, cells, endpoints, mask = s
                return (
                    out + 1,
                    cells.at[out].set(cell),
                    endpoints.at[out].set(local),
                    mask.at[out].set(True),
                )

            return jax.lax.cond(
                valid[i] & (out < capacity),
                write,
                lambda s: s,
                (out, cells, endpoints, mask),
            )

        _, cells, endpoints, mask = jax.lax.fori_loop(
            0,
            interval_starts.shape[0],
            add,
            (jnp.array(0, dtype=jnp.int32), cells, endpoints, mask),
        )
        no_piece = ~jnp.any(mask)
        cell = self.cell_index(start)
        local = self._local_coordinates(start, cell)
        cells = cells.at[0].set(jnp.where(no_piece, cell, cells[0]))
        endpoints = endpoints.at[0].set(
            jnp.where(no_piece, jnp.stack((local, local)), endpoints[0])
        )
        mask = mask.at[0].set(mask[0] | no_piece)
        return GridSegment(cells, endpoints, mask)


class HermiteGrid1D(_HermiteGrid):
    def __init__(self, x, f, d1):
        self._initialize((x,), (f, d1), 1)


class HermiteGrid2D(_HermiteGrid):
    def __init__(self, x, y, f, d1, d2):
        self._initialize((x, y), (f, d1, d2), 2)

    split_segment = _HermiteGrid._split_segment


class HermiteGrid3D(_HermiteGrid):
    def __init__(self, x, y, z, f, d1, d2, d3):
        self._initialize((x, y, z), (f, d1, d2, d3), 3)

    split_segment = _HermiteGrid._split_segment


class HermiteGrid4D(_HermiteGrid):
    def __init__(self, x, y, z, w, f, d1, d2, d3, d4):
        self._initialize((x, y, z, w), (f, d1, d2, d3, d4), 4)

    split_segment = _HermiteGrid._split_segment


class QuinticHermiteGrid1D(_HermiteGrid):
    """Quintic Hermite interpolation on a one-dimensional grid."""

    def __init__(self, x, f, d1, d2):
        self._initialize((x,), (f, d1, d2), 1, jet_size=3)


class QuinticHermiteGrid2D(_HermiteGrid):
    """Tensor-quintic Hermite interpolation on a two-dimensional grid."""

    def __init__(self, x, y, f, d1, d2, d3, d4):
        self._initialize((x, y), (f, d1, d2, d3, d4), 2, jet_size=3)

    split_segment = _HermiteGrid._split_segment


class QuinticHermiteGrid3D(_HermiteGrid):
    """Tensor-quintic Hermite interpolation on a three-dimensional grid."""

    def __init__(self, x, y, z, f, d1, d2, d3, d4, d5, d6):
        self._initialize((x, y, z), (f, d1, d2, d3, d4, d5, d6), 3, jet_size=3)

    split_segment = _HermiteGrid._split_segment


class QuinticHermiteGrid4D(_HermiteGrid):
    """Tensor-quintic Hermite interpolation on a four-dimensional grid."""

    def __init__(self, x, y, z, w, f, d1, d2, d3, d4, d5, d6, d7, d8):
        self._initialize(
            (x, y, z, w),
            (f, d1, d2, d3, d4, d5, d6, d7, d8),
            4,
            jet_size=3,
        )

    split_segment = _HermiteGrid._split_segment
