r"""1D piecewise-polynomial finite elements for interpolation.

Each element stores four independent pieces of data rather than a single
baked coefficient array:

- ``nodes``: parameter coordinates, one per node. Storage order carries no
  topological meaning by itself.
- ``connectivity``: an explicit ``(cell_count, 2)`` node-index matrix.
  ``connectivity[c] = (i, j)`` means cell ``c`` spans node ``i`` (local
  ``t=0``) to node ``j`` (local ``t=1``), with ``nodes[j] > nodes[i]``
  required. This is genuine topology data, independent of ``nodes`` array
  order, laying groundwork for non-sequential finite-element meshes.
- ``f`` (and, for C1/C2 elements, ``d1``/``d2``): the shared value/derivative
  degrees of freedom at each node, kept in their raw physical form. They are
  never converted into Bernstein coefficients; continuity across a shared
  node follows simply because neighboring cells read the same node data.
- ``interior_coefficients``: raw Bernstein/Bezier control-point coefficients
  for the degrees of freedom a cell's boundary jet does not determine (empty
  for :class:`P1C0`, :class:`P3C1`, and :class:`P5C2`, whose degree exactly
  matches their continuity order).

A cell's local polynomial is assembled on demand at evaluation time from
these four pieces via
:func:`~xbernstein.hermite._endpoint_bezier_coefficients_1d`; it is never
cached.

Vector-valued interpolation
----------------------------

``f``/``d1``/``d2`` and ``interior_coefficients`` may carry a trailing
value-shape suffix, e.g. shape ``(node_count, 3)`` for a curve in
$\mathbb{R}^3$. Scalar interpolation is simply the empty-value-shape special
case; there is no separate code path.

Continuity naming
------------------

Class names encode degree and continuity directly: ``P{degree}C{continuity}``.
``P1C0``, ``P2C0``, ``P3C0`` are C0 across cells; ``P3C1``, ``P4C1``, ``P5C1``
are C1; ``P5C2`` is C2.

This module covers interpolation only. Quadrature and weak-form assembly
live in :mod:`xbernstein.finite_element`.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from .bernstein import Bernstein
from .hermite import _endpoint_bezier_coefficients_1d


class _Element1D(eqx.Module):
    """Shared implementation for the public ``P{degree}C{continuity}`` classes."""

    nodes: jax.Array
    connectivity: jax.Array
    f: jax.Array
    interior_coefficients: jax.Array
    degree: int = eqx.field(static=True)
    continuity: int = eqx.field(static=True)

    def _build(self, nodes, connectivity, jets, degree, continuity, interior_coefficients):
        nodes = jnp.asarray(nodes)
        if nodes.ndim != 1 or nodes.shape[0] < 2:
            raise ValueError("nodes must be a one-dimensional array of length >= 2")
        node_count = nodes.shape[0]

        connectivity = jnp.asarray(connectivity)
        if connectivity.ndim != 2 or connectivity.shape[1] != 2 or connectivity.shape[0] < 1:
            raise ValueError(
                "connectivity must have shape (cell_count, 2) with cell_count >= 1"
            )
        cell_count = connectivity.shape[0]

        jet_size = continuity + 1
        interior_dof_count = degree + 1 - 2 * jet_size
        if interior_dof_count < 0:
            raise ValueError(f"degree {degree} is too small for continuity C{continuity}")

        jet_names = ("f", "d1", "d2")[:jet_size]
        if len(jets) != jet_size:
            raise TypeError(f"P{degree}C{continuity} requires {jet_size} jet array(s)")

        converted_jets = []
        value_shape = None
        for name, jet in zip(jet_names, jets):
            jet = jnp.asarray(jet)
            if jet.ndim < 1 or jet.shape[0] != node_count:
                raise ValueError(f"{name} must have a leading axis of length {node_count}")
            if value_shape is None:
                value_shape = jet.shape[1:]
            elif jet.shape[1:] != value_shape:
                raise ValueError(
                    f"{name} must have shape ({node_count}, {value_shape})"
                )
            converted_jets.append(jet)

        expected_interior_shape = (cell_count,) + value_shape + (interior_dof_count,)
        if interior_coefficients is None:
            if interior_dof_count != 0:
                raise TypeError("interior_coefficients is required for this element")
            interior_coefficients = jnp.zeros(expected_interior_shape)
        else:
            interior_coefficients = jnp.asarray(interior_coefficients)
            if interior_coefficients.shape != expected_interior_shape:
                raise ValueError(
                    f"interior_coefficients must have shape {expected_interior_shape}"
                )

        dtype = jnp.result_type(nodes, interior_coefficients, *converted_jets, jnp.float32)
        nodes = nodes.astype(dtype)
        interior_coefficients = interior_coefficients.astype(dtype)
        converted_jets = [jet.astype(dtype) for jet in converted_jets]

        connectivity = eqx.error_if(
            connectivity,
            jnp.any((connectivity < 0) | (connectivity >= node_count)),
            "connectivity entries must be valid node indices",
        )
        connectivity = connectivity.astype(jnp.int32)

        nodes = eqx.error_if(
            nodes, jnp.any(~jnp.isfinite(nodes)), "nodes must contain only finite values"
        )
        starts = jnp.take(nodes, connectivity[:, 0])
        ends = jnp.take(nodes, connectivity[:, 1])
        nodes = eqx.error_if(
            nodes,
            jnp.any(ends <= starts),
            "connectivity must reference strictly increasing node coordinates per cell",
        )
        converted_jets = [
            eqx.error_if(
                jet, jnp.any(~jnp.isfinite(jet)), f"{name} must contain only finite values"
            )
            for name, jet in zip(jet_names, converted_jets)
        ]
        interior_coefficients = eqx.error_if(
            interior_coefficients,
            jnp.any(~jnp.isfinite(interior_coefficients)),
            "interior_coefficients must contain only finite values",
        )

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "connectivity", connectivity)
        object.__setattr__(self, "interior_coefficients", interior_coefficients)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "continuity", continuity)
        for name, jet in zip(jet_names, converted_jets):
            object.__setattr__(self, name, jet)

    @property
    def _jets(self):
        return (self.f,)

    @property
    def node_count(self) -> int:
        """Return the number of stored nodes."""
        return self.nodes.shape[0]

    @property
    def cell_count(self) -> int:
        """Return the number of cells in ``connectivity``."""
        return self.connectivity.shape[0]

    @property
    def value_shape(self) -> tuple:
        """Return the trailing value shape shared by ``f``/``d1``/``d2``."""
        return self.f.shape[1:]

    @property
    def interior_dof_count(self) -> int:
        """Return the number of unshared Bezier control points per cell."""
        return self.degree + 1 - 2 * (self.continuity + 1)

    @property
    def dtype(self) -> str:
        """Return the scalar field used for the stored data."""
        return str(self.nodes.dtype)

    def cell_index(self, point) -> jax.Array:
        """Return the index of the cell whose connectivity brackets ``point``."""
        point = jnp.asarray(point, dtype=self.nodes.dtype)
        starts = jnp.take(self.nodes, self.connectivity[:, 0])
        ends = jnp.take(self.nodes, self.connectivity[:, 1])
        clipped = jnp.clip(point, jnp.min(starts), jnp.max(ends))
        contains = (clipped >= starts) & (clipped <= ends)
        return jnp.argmax(contains).astype(jnp.int32)

    def _local_coordinate(self, point, cell_index):
        node0 = self.connectivity[cell_index, 0]
        node1 = self.connectivity[cell_index, 1]
        start = self.nodes[node0]
        end = self.nodes[node1]
        clipped = jnp.clip(point, jnp.minimum(start, end), jnp.maximum(start, end))
        return jnp.clip((clipped - start) / (end - start), 0.0, 1.0)

    def cell_interpolant(self, cell_index) -> Bernstein:
        """Return the :class:`Bernstein` polynomial local to one cell."""
        cell_index = jnp.asarray(cell_index).astype(jnp.int32)
        cell_index = eqx.error_if(
            cell_index,
            (cell_index < 0) | (cell_index >= self.cell_count),
            "cell_index is outside the grid",
        )
        node0 = self.connectivity[cell_index, 0]
        node1 = self.connectivity[cell_index, 1]
        width = self.nodes[node1] - self.nodes[node0]
        jet_size = self.continuity + 1
        left_jets = tuple(
            jnp.take(jet, node0, axis=0) * width**k for k, jet in enumerate(self._jets)
        )
        right_jets = tuple(
            jnp.take(jet, node1, axis=0) * width**k for k, jet in enumerate(self._jets)
        )
        coefficients = _endpoint_bezier_coefficients_1d(
            self.degree, jet_size, left_jets, right_jets, self.interior_coefficients[cell_index]
        )
        return Bernstein(coefficients)

    def __call__(self, point):
        """Evaluate the piecewise interpolant at ``point``."""
        point = jnp.asarray(point, dtype=self.nodes.dtype)
        point = eqx.error_if(
            point, jnp.any(~jnp.isfinite(point)), "point must contain only finite values"
        )
        flat = point.reshape(-1)

        def evaluate(p):
            index = self.cell_index(p)
            local = self._local_coordinate(p, index)
            return self.cell_interpolant(index)(local)

        values = jax.vmap(evaluate)(flat)
        return values.reshape(point.shape + self.value_shape)


class _C1Element1D(_Element1D):
    d1: jax.Array

    @property
    def _jets(self):
        return (self.f, self.d1)


class _C2Element1D(_C1Element1D):
    d2: jax.Array

    @property
    def _jets(self):
        return (self.f, self.d1, self.d2)


class P1C0(_Element1D):
    """Piecewise-linear interpolation, C0 continuous across cells."""

    def __init__(self, nodes, connectivity, f):
        self._build(nodes, connectivity, (f,), degree=1, continuity=0, interior_coefficients=None)


class P2C0(_Element1D):
    """Piecewise-quadratic interpolation, C0 continuous across cells."""

    def __init__(self, nodes, connectivity, f, interior_coefficients):
        self._build(
            nodes, connectivity, (f,), degree=2, continuity=0,
            interior_coefficients=interior_coefficients,
        )


class P3C0(_Element1D):
    """Piecewise-cubic interpolation, C0 continuous across cells."""

    def __init__(self, nodes, connectivity, f, interior_coefficients):
        self._build(
            nodes, connectivity, (f,), degree=3, continuity=0,
            interior_coefficients=interior_coefficients,
        )


class P3C1(_C1Element1D):
    """Piecewise-cubic Hermite interpolation, C1 continuous across cells."""

    def __init__(self, nodes, connectivity, f, d1):
        self._build(
            nodes, connectivity, (f, d1), degree=3, continuity=1, interior_coefficients=None
        )


class P4C1(_C1Element1D):
    """Piecewise-quartic Hermite interpolation, C1 continuous across cells."""

    def __init__(self, nodes, connectivity, f, d1, interior_coefficients):
        self._build(
            nodes, connectivity, (f, d1), degree=4, continuity=1,
            interior_coefficients=interior_coefficients,
        )


class P5C1(_C1Element1D):
    """Piecewise-quintic Hermite interpolation, C1 continuous across cells."""

    def __init__(self, nodes, connectivity, f, d1, interior_coefficients):
        self._build(
            nodes, connectivity, (f, d1), degree=5, continuity=1,
            interior_coefficients=interior_coefficients,
        )


class P5C2(_C2Element1D):
    """Piecewise-quintic Hermite interpolation, C2 continuous across cells."""

    def __init__(self, nodes, connectivity, f, d1, d2):
        self._build(
            nodes, connectivity, (f, d1, d2), degree=5, continuity=2,
            interior_coefficients=None,
        )
