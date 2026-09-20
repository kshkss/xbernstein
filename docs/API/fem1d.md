# 1D finite-element interpolation

Each class in this module represents a piecewise-polynomial field on a 1D
domain, continuous across cell boundaries to a fixed order. Unlike
[Hermite grid interpolation](hermite_grid.md), these classes keep their
degrees of freedom as four independent, explicitly stored pieces rather than
one baked coefficient array:

- `nodes` — parameter coordinates, one per node. Storage order carries no
  topological meaning by itself.
- `connectivity` — an explicit `(cell_count, 2)` node-index matrix.
  `connectivity[c] = (i, j)` means cell `c` spans node `i` (local `t=0`) to
  node `j` (local `t=1`), with `nodes[j] > nodes[i]` required. Because this is
  genuine topology data rather than array adjacency, `nodes` and DOF arrays
  need not be stored in spatial order — this is deliberate groundwork for
  general (non-sequential) finite-element meshes.
- `f`, and for C1/C2 classes `d1`/`d2` — the value and derivative degrees of
  freedom shared at each node, kept in their raw physical form. They are
  **never** converted into Bernstein coefficients; continuity across a shared
  node follows simply because neighboring cells read the same node data.
- `interior_coefficients` — raw Bernstein/Bezier control-point coefficients
  for the degrees of freedom a cell's boundary data does not determine. Empty
  for `P1C0`, `P3C1`, and `P5C2`, whose degree exactly matches their
  continuity order; required for the other four classes.

A cell's local `Bernstein` polynomial is assembled on demand at evaluation
time from these four pieces — it is not cached.

## Naming

Class names encode degree and continuity directly as `P{degree}C{continuity}`:
`P1C0`, `P2C0`, `P3C0` are C0 across cells; `P3C1`, `P4C1`, `P5C1` are C1;
`P5C2` is C2. `P1C0`, `P3C1`, and `P5C2` are numerically identical, for a
single cell, to `linear_interpolate_1d`, `hermite_interpolate_1d`, and
`quintic_hermite_interpolate_1d` respectively.

## Interior degrees of freedom

For a degree-`n` element with continuity `k` (`jet_size = k + 1` shared
derivatives at each endpoint), the shared boundary jets determine
`2 * jet_size` of the `n + 1` Bezier control points; the remaining
`n + 1 - 2 * jet_size` are supplied directly as `interior_coefficients`:

| Class | degree | continuity | interior DOFs per cell |
| --- | --- | --- | --- |
| `P1C0` | 1 | 0 | 0 |
| `P2C0` | 2 | 0 | 1 |
| `P3C0` | 3 | 0 | 2 |
| `P3C1` | 3 | 1 | 0 |
| `P4C1` | 4 | 1 | 1 |
| `P5C1` | 5 | 1 | 2 |
| `P5C2` | 5 | 2 | 0 |

## Vector-valued interpolation

`f`/`d1`/`d2` and `interior_coefficients` may carry a trailing value-shape
suffix, e.g. shape `(node_count, 3)` for a curve in $\mathbb{R}^3$. Scalar
interpolation is simply the empty-value-shape special case — there is no
separate code path, since the same closed-form assembly is elementwise over
any leading batch/value axes, exactly like `Bernstein`'s `*batch order`
convention.

```python
import jax.numpy as jnp
from xbernstein import P3C1

nodes = jnp.array([0.0, 1.0, 2.0])
connectivity = jnp.array([[0, 1], [1, 2]])
f = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.5], [2.0, 0.0, 1.0]])
d1 = jnp.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.5], [1.0, 1.0, 0.0]])
curve = P3C1(nodes, connectivity, f, d1)
curve(1.5)  # shape (3,)
```

## Scope

This module covers interpolation only: point evaluation of a continuous
piecewise-polynomial field. It does not provide quadrature, mass/stiffness/
load matrices, or general mesh assembly — those live in
[`xbernstein.finite_element`](finite_element.md), which is a separate,
single-reference-element track. `connectivity` here is accepted as supplied
(shape- and index-range validated); the caller is responsible for it forming
a valid, non-overlapping domain partition.

## C0 continuity

::: xbernstein.P1C0

::: xbernstein.P2C0

::: xbernstein.P3C0

## C1 continuity

::: xbernstein.P3C1

::: xbernstein.P4C1

::: xbernstein.P5C1

## C2 continuity

::: xbernstein.P5C2
