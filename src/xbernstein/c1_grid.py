r"""C1 tensor-product Bernstein finite elements on rectilinear grids.

Continuity across a shared cell face is C1 (value and, in the crossed
axis's direction, first derivative), by construction from shared vertex
jets, exactly like :class:`~xbernstein.hermite_grid.HermiteGrid1D`-`4D`
(which is precisely the degree-3 case of this family). Unlike C0
continuity (:mod:`xbernstein.c0_grid`), matching a derivative across a cell
boundary is width-dependent, so a plain shared-array-slice trick is not
enough on its own for the axis being crossed — the near-boundary control
points must be computed, per cell, from the shared vertex jets via the
standard Hermite endpoint-conversion identity (see
:func:`xbernstein.hermite._boundary_jet_bezier_coefficients_nd`).

Jet size is fixed at 2 (value + first derivative, plus all mixed partials
up to total order ``dimension`` — the same vertex data
:class:`~xbernstein.hermite_grid.HermiteGrid1D`-`4D` already requires)
regardless of degree. For degree 3, jets fully determine every control
point (``interior_dof_count = 4 - 2*2 = 0``); for degree 4/5, they leave 1
or 2 free control points per axis, exactly like
:class:`~xbernstein.fem1d.P4C1`/:class:`~xbernstein.fem1d.P5C1` in 1D. Those
free positions are stored in one dense ``interior_coefficients`` array
shaped like the full refined grid (not a packed per-cell array): a
position is either jet-determined or read from ``interior_coefficients``,
never both, so two cells sharing a face automatically agree — there is
exactly one array slot per global position, regardless of whether that
slot sits on the axis being crossed or is merely tangentially shared.

Like :mod:`xbernstein.c0_grid`, one dense ``coefficients`` array lives on a
*refined* structured grid: along axis ``a``, every original cell is
subdivided into ``degree`` equal parameter steps
(``dof_shape[a] = degree * cell_count[a] + 1``), and a cell's local
``(degree + 1) ** dimension`` control block is an overlapping window of
that array (stride ``degree``, window size ``degree + 1``). Point
location, windowing, and segment splitting are copied verbatim from
:class:`xbernstein.c0_grid._C0Grid`, which only depends on ``axes`` and a
``coefficients`` array of the right shape.
"""

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp

from .bernstein import Bernstein
from .bernstein_2d import Bernstein2D
from .bernstein_3d import Bernstein3D
from .bernstein_4d import Bernstein4D
from .hermite import _boundary_jet_bezier_coefficients_nd, _multi_indices
from .hermite_grid import GridSegment

_JET_SIZE = 2


class _C1Grid(eqx.Module):
    """Shared implementation for the public ``P{degree}C1Grid{dimension}D`` classes."""

    axes: tuple
    coefficients: jax.Array
    dimension: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)

    def _initialize(self, axes, groups, dimension, degree, interior_coefficients):
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
        cell_counts = tuple(size - 1 for size in grid_shape)
        dof_shape = tuple(degree * count + 1 for count in cell_counts)
        interior_dof_count = degree + 1 - 2 * _JET_SIZE

        values = tuple(jnp.asarray(group) for group in groups)
        expected_group_count = (_JET_SIZE - 1) * dimension + 1
        if len(values) != expected_group_count:
            raise TypeError(
                f"P{degree}C1Grid{dimension}D requires {expected_group_count} "
                "derivative groups"
            )
        if values[0].ndim < dimension or values[0].shape[-dimension:] != grid_shape:
            raise ValueError(f"f must end with grid shape {grid_shape}")
        batch_shape = values[0].shape[:-dimension]
        expected_counts = tuple(
            len(_multi_indices(dimension, _JET_SIZE, total))
            for total in range(1, dimension + 1)
        )
        for total, (group, count) in enumerate(zip(values[1:], expected_counts), start=1):
            expected = batch_shape + grid_shape + (() if count == 1 else (count,))
            if group.shape != expected:
                raise ValueError(f"d{total} must have shape {expected}")

        expected_interior_shape = batch_shape + dof_shape
        if interior_dof_count > 0:
            if interior_coefficients is None:
                raise TypeError("interior_coefficients is required for this element")
            interior_coefficients = jnp.asarray(interior_coefficients)
            if interior_coefficients.shape != expected_interior_shape:
                raise ValueError(
                    f"interior_coefficients must have shape {expected_interior_shape}"
                )
        else:
            interior_coefficients = jnp.zeros(expected_interior_shape)

        dtype = jnp.result_type(
            *values, interior_coefficients, *coordinates, jnp.float32
        )
        derivative_names = ("f",) + tuple(
            f"d{i}" for i in range(1, expected_group_count)
        )
        converted = []
        for name, group in zip(derivative_names, values):
            group = group.astype(dtype)
            group = eqx.error_if(
                group,
                jnp.any(~jnp.isfinite(group)),
                f"{name} must contain only finite values",
            )
            converted.append(group)
        coordinates = tuple(axis.astype(dtype) for axis in coordinates)
        interior_coefficients = interior_coefficients.astype(dtype)
        interior_coefficients = eqx.error_if(
            interior_coefficients,
            jnp.any(~jnp.isfinite(interior_coefficients)),
            "interior_coefficients must contain only finite values",
        )

        coefficients = self._build_coefficients(
            coordinates,
            tuple(converted),
            interior_coefficients,
            dimension,
            degree,
            batch_shape,
            cell_counts,
        )
        object.__setattr__(self, "axes", coordinates)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "degree", degree)

    @staticmethod
    def _build_coefficients(
        coordinates, groups, interior_coefficients, dimension, degree, batch_shape, cell_counts
    ):
        """Combine shared vertex jets and raw interior data into one refined array."""
        batch_dimensions = len(batch_shape)
        vertices = tuple(itertools.product((0, 1), repeat=dimension))

        def gather_cell_vertices(group):
            tail = group.shape[batch_dimensions + dimension :]
            cell_vertices = []
            for vertex in vertices:
                slices = (slice(None),) * batch_dimensions + tuple(
                    slice(offset, offset + size)
                    for offset, size in zip(vertex, cell_counts)
                )
                cell_vertices.append(group[slices])
            stacked = jnp.stack(cell_vertices, axis=batch_dimensions + dimension)
            return stacked.reshape(batch_shape + cell_counts + (2,) * dimension + tail)

        width_fields = []
        for axis_index, axis in enumerate(coordinates):
            width_shape = [1] * dimension
            width_shape[axis_index] = cell_counts[axis_index]
            width_fields.append(jnp.diff(axis).reshape(width_shape))

        local_groups = [gather_cell_vertices(groups[0])]
        for total, source in enumerate(groups[1:], start=1):
            group = gather_cell_vertices(source)
            alphas = _multi_indices(dimension, _JET_SIZE, total)
            scales = [
                jnp.prod(
                    jnp.stack(
                        [
                            jnp.broadcast_to(width, cell_counts) ** order
                            for width, order in zip(width_fields, alpha)
                        ]
                    ),
                    axis=0,
                )
                for alpha in alphas
            ]
            if len(alphas) == 1:
                scale = scales[0].reshape(
                    (1,) * batch_dimensions + cell_counts + (1,) * dimension
                )
            else:
                scale = jnp.stack(scales, axis=-1).reshape(
                    (1,) * batch_dimensions
                    + cell_counts
                    + (1,) * dimension
                    + (len(alphas),)
                )
            local_groups.append(group * scale)

        jet_coefficients, boundary_mask = _boundary_jet_bezier_coefficients_nd(
            dimension, degree, _JET_SIZE, tuple(local_groups)
        )
        # jet_coefficients: batch_shape + cell_counts + (degree + 1,) * dimension
        # boundary_mask: (degree + 1,) * dimension (no batch axes)

        total_cells = 1
        for count in cell_counts:
            total_cells *= count
        block_shape = (degree + 1,) * dimension
        jet_flat = jet_coefficients.reshape(
            batch_shape + (total_cells,) + block_shape
        )

        def scatter_cell(flat_index, coefficients):
            cell_multi_index = jnp.unravel_index(flat_index, cell_counts)
            block = jnp.take(jet_flat, flat_index, axis=batch_dimensions)
            starts = tuple(index * degree for index in cell_multi_index)
            full_starts = (0,) * batch_dimensions + starts
            current = jax.lax.dynamic_slice(
                coefficients, full_starts, batch_shape + block_shape
            )
            updated = jnp.where(boundary_mask, block, current)
            return jax.lax.dynamic_update_slice(coefficients, updated, full_starts)

        return jax.lax.fori_loop(0, total_cells, scatter_cell, interior_coefficients)

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
    def shape(self):
        """Return the leading batch/value shape shared by the jet groups."""
        return self.coefficients.shape[: -self.dimension]

    @property
    def dtype(self) -> str:
        """Return the scalar dtype used by the stored control points."""
        return str(self.coefficients.dtype)

    @property
    def grid_shape(self):
        """Return the number of original grid vertices along each axis."""
        return tuple(axis.shape[0] for axis in self.axes)

    @property
    def dof_shape(self):
        """Return the shape of the refined control-point grid."""
        return self.coefficients.shape[-self.dimension :]

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
        window = self.degree + 1
        for offset in range(self.dimension):
            axis_position = batch_dimensions + offset
            start = cell_index[offset] * self.degree
            result = jax.lax.dynamic_slice_in_dim(result, start, window, axis=axis_position)
        return result

    def cell_interpolant(self, cell_index):
        """Return the :class:`Bernstein`-family polynomial local to one cell."""
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
        """Evaluate the piecewise interpolant at ``point``."""
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


class P3C1Grid1D(_C1Grid):
    """Cubic C1 interpolation on a one-dimensional rectilinear grid."""

    def __init__(self, x, f, d1):
        self._initialize((x,), (f, d1), dimension=1, degree=3, interior_coefficients=None)


class P4C1Grid1D(_C1Grid):
    """Quartic C1 interpolation on a one-dimensional rectilinear grid."""

    def __init__(self, x, f, d1, interior_coefficients):
        self._initialize(
            (x,), (f, d1), dimension=1, degree=4, interior_coefficients=interior_coefficients
        )


class P5C1Grid1D(_C1Grid):
    """Quintic C1 interpolation on a one-dimensional rectilinear grid."""

    def __init__(self, x, f, d1, interior_coefficients):
        self._initialize(
            (x,), (f, d1), dimension=1, degree=5, interior_coefficients=interior_coefficients
        )


class P3C1Grid2D(_C1Grid):
    """Bicubic C1 interpolation on a two-dimensional rectilinear grid."""

    def __init__(self, x, y, f, d1, d2):
        self._initialize((x, y), (f, d1, d2), dimension=2, degree=3, interior_coefficients=None)

    split_segment = _C1Grid._split_segment


class P4C1Grid2D(_C1Grid):
    """Biquartic C1 interpolation on a two-dimensional rectilinear grid."""

    def __init__(self, x, y, f, d1, d2, interior_coefficients):
        self._initialize(
            (x, y), (f, d1, d2), dimension=2, degree=4, interior_coefficients=interior_coefficients
        )

    split_segment = _C1Grid._split_segment


class P5C1Grid2D(_C1Grid):
    """Biquintic C1 interpolation on a two-dimensional rectilinear grid."""

    def __init__(self, x, y, f, d1, d2, interior_coefficients):
        self._initialize(
            (x, y), (f, d1, d2), dimension=2, degree=5, interior_coefficients=interior_coefficients
        )

    split_segment = _C1Grid._split_segment


class P3C1Grid3D(_C1Grid):
    """Tricubic C1 interpolation on a three-dimensional rectilinear grid."""

    def __init__(self, x, y, z, f, d1, d2, d3):
        self._initialize(
            (x, y, z), (f, d1, d2, d3), dimension=3, degree=3, interior_coefficients=None
        )

    split_segment = _C1Grid._split_segment


class P4C1Grid3D(_C1Grid):
    """Triquartic C1 interpolation on a three-dimensional rectilinear grid."""

    def __init__(self, x, y, z, f, d1, d2, d3, interior_coefficients):
        self._initialize(
            (x, y, z),
            (f, d1, d2, d3),
            dimension=3,
            degree=4,
            interior_coefficients=interior_coefficients,
        )

    split_segment = _C1Grid._split_segment


class P5C1Grid3D(_C1Grid):
    """Triquintic C1 interpolation on a three-dimensional rectilinear grid."""

    def __init__(self, x, y, z, f, d1, d2, d3, interior_coefficients):
        self._initialize(
            (x, y, z),
            (f, d1, d2, d3),
            dimension=3,
            degree=5,
            interior_coefficients=interior_coefficients,
        )

    split_segment = _C1Grid._split_segment


class P3C1Grid4D(_C1Grid):
    """Quadricubic C1 interpolation on a four-dimensional rectilinear grid."""

    def __init__(self, x, y, z, w, f, d1, d2, d3, d4):
        self._initialize(
            (x, y, z, w),
            (f, d1, d2, d3, d4),
            dimension=4,
            degree=3,
            interior_coefficients=None,
        )

    split_segment = _C1Grid._split_segment


class P4C1Grid4D(_C1Grid):
    """Quadriquartic C1 interpolation on a four-dimensional rectilinear grid."""

    def __init__(self, x, y, z, w, f, d1, d2, d3, d4, interior_coefficients):
        self._initialize(
            (x, y, z, w),
            (f, d1, d2, d3, d4),
            dimension=4,
            degree=4,
            interior_coefficients=interior_coefficients,
        )

    split_segment = _C1Grid._split_segment


class P5C1Grid4D(_C1Grid):
    """Quadriquintic C1 interpolation on a four-dimensional rectilinear grid."""

    def __init__(self, x, y, z, w, f, d1, d2, d3, d4, interior_coefficients):
        self._initialize(
            (x, y, z, w),
            (f, d1, d2, d3, d4),
            dimension=4,
            degree=5,
            interior_coefficients=interior_coefficients,
        )

    split_segment = _C1Grid._split_segment
