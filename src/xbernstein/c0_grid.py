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

import itertools
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Shaped, jaxtyped

from .bernstein import Bernstein, _minimize
from .bernstein_2d import Bernstein2D, _minimize as _minimize_2d
from .bernstein_3d import Bernstein3D, _minimize as _minimize_3d
from .bernstein_4d import Bernstein4D, _minimize as _minimize_4d
from .hermite_grid import GridSegment


class GridOptimizeResult(NamedTuple):
    r"""Result of :meth:`_C0Grid.minimize`/:meth:`_C0Grid.maximize`.

    Unlike :class:`~xbernstein.OptimizeResult`, this also reports which cell
    the extremum was found in, since a grid's global extremum is the best of
    many independent per-cell searches. ``f`` and ``cell`` share the leading
    batch shape; ``x`` and ``cell`` share the same trailing ``(dimension,)``
    shape, and ``cell`` is the index :meth:`_C0Grid.cell_interpolant` expects
    for the cell containing ``x``.
    """

    f: jax.Array
    x: jax.Array
    cell: jax.Array


@jaxtyped(typechecker=beartype)
class _C0Grid(eqx.Module):
    """Shared implementation for the public ``P{degree}C0Grid{dimension}D`` classes."""

    axes: tuple
    coefficients: jax.Array
    dimension: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)

    @jaxtyped(typechecker=beartype)
    def _initialize(
        self,
        axes: tuple[Shaped[jax.Array, " _"], ...],
        f: Shaped[jax.Array, "..."],
        dimension: int,
        degree: int,
    ):
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")

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

    def _cell_physical_point(self, cell_index, local):
        """Invert :meth:`_local_coordinates`: map a cell-local point back to physical space."""
        starts = jnp.stack([axis[index] for axis, index in zip(self.axes, cell_index)])
        ends = jnp.stack(
            [axis[index + 1] for axis, index in zip(self.axes, cell_index)]
        )
        return starts + local * (ends - starts)

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

    _cell_solvers = {1: _minimize, 2: _minimize_2d, 3: _minimize_3d, 4: _minimize_4d}

    def _extreme(self, sign: float, max_steps: int, eps: float):
        r"""Shared branch-and-bound reduction behind :meth:`minimize`/:meth:`maximize`.

        Every cell's :meth:`cell_interpolant` is an independent tensor-product
        Bernstein polynomial, so the grid's global extremum is the best of
        each cell's *local* extremum — the convex-hull property that
        certifies branch and bound within one cell says nothing about other
        cells. This runs the same per-cell solver used by the module-level
        :func:`xbernstein.minimize` (``sign=1.0``) on every cell and keeps
        the smallest result, mirroring how :func:`xbernstein.maximize`
        negates the coefficients before minimizing (``sign=-1.0``) — so a
        single reduction (``<``) is correct for both directions, and only
        the reported value needs the sign undone at the end.

        Cost scales with the total number of cells (the product of
        per-axis cell counts), since a global extremum genuinely requires
        checking every cell. ``max_steps`` and ``eps`` must stay concrete
        Python values, as for the per-cell solvers themselves — this method
        cannot be nested inside an outer ``jax.jit``.
        """
        solver = self._cell_solvers[self.dimension]
        cell_counts = tuple(axis.shape[0] - 1 for axis in self.axes)
        batch_dimensions = len(self.shape)
        batch_shape = self.shape

        best_value = None
        best_point = None
        best_cell = None
        for cell in itertools.product(*(range(count) for count in cell_counts)):
            cell_index = jnp.asarray(cell, dtype=jnp.int32)
            coefficients = sign * self._cell_coefficients(cell_index)

            if batch_dimensions:
                flat = coefficients.reshape(
                    (-1,) + coefficients.shape[batch_dimensions:]
                )
                value, x = jax.vmap(solver, in_axes=(0, None, None))(
                    flat, max_steps, eps
                )
            else:
                value, x = solver(coefficients, max_steps, eps)

            if self.dimension == 1:
                x = x[..., None]
            if batch_dimensions:
                value = value.reshape(batch_shape)
                x = x.reshape(batch_shape + (self.dimension,))

            starts = jnp.stack([axis[index] for axis, index in zip(self.axes, cell)])
            ends = jnp.stack([axis[index + 1] for axis, index in zip(self.axes, cell)])
            point = starts + x * (ends - starts)
            cell_broadcast = jnp.broadcast_to(
                cell_index, batch_shape + (self.dimension,)
            )

            if best_value is None:
                best_value, best_point, best_cell = value, point, cell_broadcast
            else:
                better = value < best_value
                best_value = jnp.where(better, value, best_value)
                best_point = jnp.where(better[..., None], point, best_point)
                best_cell = jnp.where(better[..., None], cell_broadcast, best_cell)

        return GridOptimizeResult(f=sign * best_value, x=best_point, cell=best_cell)

    def minimize(self, max_steps: int = 200, eps: float = 1e-6):
        r"""Approximate $\min_{\mathbf{x}}p(\mathbf{x})$ over the grid's physical domain.

        Applies :func:`xbernstein.minimize`'s per-cell branch and bound to
        every cell of the grid and returns the best result as a
        :class:`GridOptimizeResult`, whose ``x`` is a *physical* point —
        unlike the module-level function's ``[0,1]``-parameter location.
        ``x`` always has trailing shape ``(dimension,)``, including for
        one-dimensional grids, matching every other physical point in this
        class's API (``__call__``, ``cell_index``, ...). ``cell`` is the
        index of the cell the extremum was found in — pass it directly to
        :meth:`cell_interpolant` — and, when several cells tie exactly
        (e.g. a degree-1 grid's extremum sitting on a shared vertex), it is
        whichever tied cell the per-cell search happens to visit first.

        Available in every dimension, unlike :meth:`split_segment`,
        :meth:`segment_grid`, and :meth:`integrate_out`.
        """
        return self._extreme(1.0, max_steps, eps)

    def maximize(self, max_steps: int = 200, eps: float = 1e-6):
        r"""Approximate $\max_{\mathbf{x}}p(\mathbf{x})$ over the grid's physical domain.

        Implemented, like :func:`xbernstein.maximize`, as the negated
        :meth:`minimize`. See :meth:`minimize` for the meaning of ``x`` and
        ``cell``.
        """
        return self._extreme(-1.0, max_steps, eps)

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
        valid = (
            jnp.isfinite(interval_starts)
            & jnp.isfinite(interval_ends)
            & (interval_ends > interval_starts)
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

        Built on :meth:`_split_segment`'s fixed-capacity decomposition: for
        every one of its ``segment_capacity`` slots (real crossing or
        padding alike), the tensor-product cell polynomial's
        :meth:`~xbernstein._tensor_bernstein._TensorBernstein.segment` gives
        the exact degree-``dimension * degree`` Bernstein restriction over
        that slot's local interval. Concatenating all ``segment_capacity``
        slots' control points (sharing the one coefficient at each internal
        breakpoint, since a restriction's first/last control points equal
        the true function value there) yields a :class:`C0Grid1D` that
        reproduces ``self`` exactly along the segment.

        Because the slot count is always ``segment_capacity`` — the static
        worst-case number of grid-line crossings for this grid — regardless
        of how many crossings ``start``/``end`` actually produce, the
        result's shape never depends on runtime values, so, unlike the
        earlier implementation, this method **is** ``jax.jit``-traceable.
        The tradeoff is size: the result is always ``segment_capacity``
        cells, not exactly as many as the segment actually crosses.

        Padding slots (past the real crossings) default to a degenerate
        zero-length interval at this grid's ``(0, ..., 0)`` corner, which
        would otherwise show up as a discontinuity exactly at the segment's
        true endpoint (`` t = jnp.linalg.norm(end - start)``) — the point
        callers are most likely to query. Every padding slot's control
        points are therefore overwritten with the constant value carried
        forward from the last real slot, so the padded tail is a flat, C0
        continuation of the true endpoint value rather than that unrelated
        corner value.

        The returned grid's ``x`` axis is the Euclidean distance travelled
        from ``start``: real slots keep their true physical length, and it
        starts at ``0`` and reaches ``jnp.linalg.norm(end - start)`` exactly
        at the last real slot's boundary — the only region meaningful to
        query. Padding slots get an arbitrary placeholder length (enough to
        keep ``x`` strictly increasing, as :class:`C0Grid1D` requires) and
        extend `x` past that point purely so the grid remains well-formed.
        """
        start = self._point(start)
        end = self._point(end)
        length = jnp.linalg.norm(end - start)
        length = eqx.error_if(length, length <= 0.0, "start and end must not coincide")

        segment = self._split_segment(start, end)
        cells, endpoints, mask = (
            segment.cell_indices,
            segment.local_endpoints,
            segment.valid_mask,
        )

        def per_slot(cell, local_pair):
            local_start, local_end = local_pair[0], local_pair[1]
            coefficients = self.cell_interpolant(cell).segment(local_start, local_end).c
            physical_start = self._cell_physical_point(cell, local_start)
            physical_end = self._cell_physical_point(cell, local_end)
            piece_length = jnp.linalg.norm(physical_end - physical_start)
            return coefficients, piece_length

        pieces, piece_lengths = jax.vmap(per_slot)(cells, endpoints)

        effective_lengths = jnp.where(mask, piece_lengths, jnp.ones_like(piece_lengths))
        x = jnp.concatenate(
            [
                jnp.zeros((1,), dtype=effective_lengths.dtype),
                jnp.cumsum(effective_lengths),
            ]
        )

        def weld(carry, slot):
            piece, valid = slot
            filled = jnp.where(valid, piece, carry[..., None])
            return filled[..., -1], filled

        _, tail = jax.lax.scan(weld, pieces[0, ..., -1], (pieces[1:], mask[1:]))
        welded_tail = jnp.moveaxis(tail[..., 1:], 0, -2).reshape(
            pieces.shape[1:-1] + (-1,)
        )
        f = jnp.concatenate([pieces[0], welded_tail], axis=-1)
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
