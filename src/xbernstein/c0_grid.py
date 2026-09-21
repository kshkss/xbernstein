r"""C0 tensor-product Bernstein finite elements on rectilinear grids.

Each cell's local polynomial is a degree-``p`` tensor-product Bernstein patch
with ``(p + 1) ** dimension`` control points. Continuity across a shared cell
face requires the *entire* boundary control-point sub-array on that face to
match between the two neighboring cells, not just the corner values — a
degree-``p`` Bezier curve/patch restricted to a face has ``p - 1`` more
degrees of freedom along that face than its two/four corners alone fix.

This module stores one dense array of control points on a *refined*
structured grid: along each axis, every original grid cell is subdivided into
``p`` equal parameter steps, so axis ``a`` contributes
``p * (len(axis_a) - 1) + 1`` positions. A cell's local ``(p + 1) **
dimension`` control-point block is simply an overlapping window of that array
(stride ``p``, window size ``p + 1`` per axis); neighboring cells' windows
share their common boundary slice exactly, since it is the same underlying
array data. No conversion step is needed at construction time.

Only entries that land on an *original* grid vertex (index 0 or ``p`` along
every axis) equal the true function value there — that is a general property
of Bernstein control points at a domain corner. Every other stored entry is a
raw shared Bernstein coefficient, not a nodal function value, matching the
"freedoms are Bernstein coefficients, not Lagrange values" convention used
elsewhere in this library.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from .bernstein import Bernstein
from .bernstein_2d import Bernstein2D
from .bernstein_3d import Bernstein3D
from .bernstein_4d import Bernstein4D
from .hermite_grid import GridSegment


class _C0Grid(eqx.Module):
    """Shared implementation for the public ``P{degree}C0Grid{dimension}D`` classes."""

    axes: tuple
    coefficients: jax.Array
    dimension: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)

    def _initialize(self, axes, f, dimension, degree):
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

        cell_counts = tuple(axis.shape[0] - 1 for axis in coordinates)
        dof_shape = tuple(degree * count + 1 for count in cell_counts)

        f = jnp.asarray(f)
        if f.ndim < dimension or f.shape[-dimension:] != dof_shape:
            raise ValueError(f"f must end with shape {dof_shape}")

        dtype = jnp.result_type(f, *coordinates, jnp.float32)
        f = f.astype(dtype)
        f = eqx.error_if(
            f, jnp.any(~jnp.isfinite(f)), "f must contain only finite values"
        )
        coordinates = tuple(axis.astype(dtype) for axis in coordinates)

        object.__setattr__(self, "axes", coordinates)
        object.__setattr__(self, "coefficients", f)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "degree", degree)

    @property
    def f(self):
        """Return the stored control-point array on the refined grid."""
        return self.coefficients

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
        """Return the leading batch/value shape shared by ``f``."""
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
            result = jax.lax.dynamic_slice_in_dim(
                result, start, window, axis=axis_position
            )
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

    def _segment_parameters(self, start, end):
        """Return sorted per-axis grid-line crossing parameters along a segment.

        ``start`` and ``end`` are physical points; the result gives every
        interval of the parameter $t\\in[0,1]$ (with $t=0$ at ``start`` and
        $t=1$ at ``end``) between consecutive grid-line crossings, along with
        a mask of which intervals have positive width.
        """
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
        return interval_starts, interval_ends, valid

    def _split_segment(self, start, end):
        start = self._point(start)
        end = self._point(end)
        delta = end - start
        interval_starts, interval_ends, valid = self._segment_parameters(start, end)
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

    def _segment_grid(self, start, end):
        r"""Restrict the interpolant to a straight segment as a 1D C0 grid.

        The segment is split into pieces at every grid-line crossing, as in
        :meth:`_split_segment`. On each piece, the tensor-product cell
        polynomial's :meth:`~xbernstein._tensor_bernstein._TensorBernstein.segment`
        gives the exact degree-``dimension * degree`` Bernstein restriction
        along that piece. Concatenating the pieces' control points yields a
        :class:`C0Grid1D` that reproduces ``self`` exactly along the segment
        — this relies on a restriction's first and last control points
        equalling the true function value at the piece's endpoints, so
        consecutive pieces already share the one coefficient at their
        common breakpoint.

        This method is not JIT-traceable: the number of pieces depends on
        where the segment happens to cross grid lines, so the result's shape
        is determined only once ``start`` and ``end`` are concrete.

        The returned grid's ``x`` axis is the Euclidean distance travelled
        from ``start``, so it starts at ``0`` and ends at
        ``jnp.linalg.norm(end - start)``.
        """
        start = self._point(start)
        end = self._point(end)
        delta = end - start
        length = jnp.linalg.norm(delta)
        if not bool(length > 0.0):
            raise ValueError("start and end must not coincide")

        interval_starts, interval_ends, valid = self._segment_parameters(start, end)
        interval_starts = interval_starts[valid]
        interval_ends = interval_ends[valid]

        pieces = []
        for piece_start, piece_end in zip(interval_starts, interval_ends):
            physical_start = start + piece_start * delta
            physical_end = start + piece_end * delta
            cell = self.cell_index(0.5 * (physical_start + physical_end))
            local_start = self._local_coordinates(physical_start, cell)
            local_end = self._local_coordinates(physical_end, cell)
            polynomial = self.cell_interpolant(cell).segment(local_start, local_end)
            pieces.append(polynomial.c)

        breakpoints = jnp.stack([interval_starts[0], *interval_ends])
        x = breakpoints * length
        f = jnp.concatenate([pieces[0]] + [piece[1:] for piece in pieces[1:]])
        return C0Grid1D(x, f, degree=self.dimension * self.degree)

    def _integrate_out(self, axis: int = 0):
        r"""Integrate the represented function over one axis' full physical range.

        Because the tensor-product Bernstein basis factorizes across axes,
        the exact unit-parameter integral along ``axis`` reduces to
        ``sum(window, axis) / (degree + 1)`` regardless of what the other
        coefficient-array axes represent (see
        :meth:`~xbernstein._tensor_bernstein._TensorBernstein.integrate_out`).
        Applying that reduction to each physical cell's window along
        ``axis`` (scaled by that cell's physical width) and summing over
        cells gives the exact definite integral over the axis' full
        physical range, directly on the shared refined array — the
        surviving axes' overlapping-window structure is untouched, so C0
        continuity along them is preserved automatically.

        Returns a ``(dimension - 1)``-dimensional grid of the same
        ``degree`` over the remaining axes, in their original order. Not
        available on a 1D grid: integrating out its sole axis would yield a
        scalar, not a grid.

        Unlike :meth:`_segment_grid`, this method's output shape does not
        depend on runtime values (``self.grid_shape`` is static), so it is
        ``jax.jit``/``jax.vmap``-traceable.
        """
        if not 0 <= axis < self.dimension:
            raise ValueError(f"axis must be in [0, {self.dimension}), got {axis}")

        batch_dimensions = len(self.shape)
        axis_position = batch_dimensions + axis
        degree = self.degree
        cell_count = self.grid_shape[axis] - 1
        widths = jnp.diff(self.axes[axis])

        accumulator = 0.0
        for c in range(cell_count):
            window = jax.lax.slice_in_dim(
                self.coefficients,
                c * degree,
                c * degree + degree + 1,
                axis=axis_position,
            )
            accumulator = accumulator + widths[c] * jnp.sum(
                window, axis=axis_position
            ) / (degree + 1)

        new_axes = self.axes[:axis] + self.axes[axis + 1 :]
        grid_types = {
            1: (P1C0Grid1D, P2C0Grid1D, P3C0Grid1D),
            2: (P1C0Grid2D, P2C0Grid2D, P3C0Grid2D),
            3: (P1C0Grid3D, P2C0Grid3D, P3C0Grid3D),
        }
        cls = grid_types[self.dimension - 1][degree - 1]
        return cls(*new_axes, accumulator)


class C0Grid1D(_C0Grid):
    """Piecewise C0 interpolation of an arbitrary degree on a 1D rectilinear grid.

    Unlike :class:`P1C0Grid1D`–:class:`P3C0Grid1D`, ``degree`` is a
    constructor argument rather than fixed by the class. This is what lets
    :meth:`_C0Grid.segment_grid` return an exact restriction whose degree
    (``dimension * degree`` of the source grid) can exceed 3.
    """

    def __init__(self, x, f, degree):
        self._initialize((x,), f, dimension=1, degree=degree)


class P1C0Grid1D(_C0Grid):
    """Piecewise-linear C0 interpolation on a one-dimensional rectilinear grid."""

    def __init__(self, x, f):
        self._initialize((x,), f, dimension=1, degree=1)


class P2C0Grid1D(_C0Grid):
    """Piecewise-quadratic C0 interpolation on a one-dimensional rectilinear grid."""

    def __init__(self, x, f):
        self._initialize((x,), f, dimension=1, degree=2)


class P3C0Grid1D(_C0Grid):
    """Piecewise-cubic C0 interpolation on a one-dimensional rectilinear grid."""

    def __init__(self, x, f):
        self._initialize((x,), f, dimension=1, degree=3)


class P1C0Grid2D(_C0Grid):
    """Bilinear C0 interpolation on a two-dimensional rectilinear grid."""

    def __init__(self, x, y, f):
        self._initialize((x, y), f, dimension=2, degree=1)

    split_segment = _C0Grid._split_segment
    segment_grid = _C0Grid._segment_grid
    integrate_out = _C0Grid._integrate_out


class P2C0Grid2D(_C0Grid):
    """Biquadratic C0 interpolation on a two-dimensional rectilinear grid."""

    def __init__(self, x, y, f):
        self._initialize((x, y), f, dimension=2, degree=2)

    split_segment = _C0Grid._split_segment
    segment_grid = _C0Grid._segment_grid
    integrate_out = _C0Grid._integrate_out


class P3C0Grid2D(_C0Grid):
    """Bicubic C0 interpolation on a two-dimensional rectilinear grid."""

    def __init__(self, x, y, f):
        self._initialize((x, y), f, dimension=2, degree=3)

    split_segment = _C0Grid._split_segment
    segment_grid = _C0Grid._segment_grid
    integrate_out = _C0Grid._integrate_out


class P1C0Grid3D(_C0Grid):
    """Trilinear C0 interpolation on a three-dimensional rectilinear grid."""

    def __init__(self, x, y, z, f):
        self._initialize((x, y, z), f, dimension=3, degree=1)

    split_segment = _C0Grid._split_segment
    segment_grid = _C0Grid._segment_grid
    integrate_out = _C0Grid._integrate_out


class P2C0Grid3D(_C0Grid):
    """Triquadratic C0 interpolation on a three-dimensional rectilinear grid."""

    def __init__(self, x, y, z, f):
        self._initialize((x, y, z), f, dimension=3, degree=2)

    split_segment = _C0Grid._split_segment
    segment_grid = _C0Grid._segment_grid
    integrate_out = _C0Grid._integrate_out


class P3C0Grid3D(_C0Grid):
    """Tricubic C0 interpolation on a three-dimensional rectilinear grid."""

    def __init__(self, x, y, z, f):
        self._initialize((x, y, z), f, dimension=3, degree=3)

    split_segment = _C0Grid._split_segment
    segment_grid = _C0Grid._segment_grid
    integrate_out = _C0Grid._integrate_out


class P1C0Grid4D(_C0Grid):
    """Quadrilinear C0 interpolation on a four-dimensional rectilinear grid."""

    def __init__(self, x, y, z, w, f):
        self._initialize((x, y, z, w), f, dimension=4, degree=1)

    split_segment = _C0Grid._split_segment
    segment_grid = _C0Grid._segment_grid
    integrate_out = _C0Grid._integrate_out


class P2C0Grid4D(_C0Grid):
    """Quadriquadratic C0 interpolation on a four-dimensional rectilinear grid."""

    def __init__(self, x, y, z, w, f):
        self._initialize((x, y, z, w), f, dimension=4, degree=2)

    split_segment = _C0Grid._split_segment
    segment_grid = _C0Grid._segment_grid
    integrate_out = _C0Grid._integrate_out


class P3C0Grid4D(_C0Grid):
    """Quadricubic C0 interpolation on a four-dimensional rectilinear grid."""

    def __init__(self, x, y, z, w, f):
        self._initialize((x, y, z, w), f, dimension=4, degree=3)

    split_segment = _C0Grid._split_segment
    segment_grid = _C0Grid._segment_grid
    integrate_out = _C0Grid._integrate_out
