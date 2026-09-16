# C1 tensor-product Bernstein grids

## One-dimensional

::: xbernstein.P3C1Grid1D

::: xbernstein.P4C1Grid1D

::: xbernstein.P5C1Grid1D

## Two-dimensional

::: xbernstein.P3C1Grid2D

::: xbernstein.P4C1Grid2D

::: xbernstein.P5C1Grid2D

## Three-dimensional

::: xbernstein.P3C1Grid3D

::: xbernstein.P4C1Grid3D

::: xbernstein.P5C1Grid3D

## Four-dimensional

::: xbernstein.P3C1Grid4D

::: xbernstein.P4C1Grid4D

::: xbernstein.P5C1Grid4D

Each class shares vertex jets — value plus every mixed partial derivative up
to total order `dimension`, the same data
[`HermiteGrid1D`–`4D`](hermite_grid.md) already requires — and builds a
tensor-product Bernstein cell from them. `P3C1Grid{dimension}D` is exactly
`HermiteGrid{dimension}D`'s degree-3 case (no leftover freedom: the jets
fully determine every control point), generalized through the same
machinery as `P4C1Grid{dimension}D`/`P5C1Grid{dimension}D`, which leave 1 or
2 free control points per axis.

Unlike [C0 grids](c0_grid.md), matching a *derivative* across a shared cell
face is width-dependent, so the near-boundary control points in the axis
being crossed are computed per cell from the shared vertex jets (the
standard Hermite endpoint-conversion identity), not merely copied. The
leftover free control points (for degree 4/5) live in one dense
`interior_coefficients` array shaped like the full refined grid, not a
packed per-cell array: a position is either jet-determined or read from
`interior_coefficients`, never both, so a position tangentially shared
between two cells (interior along one axis, but sitting on a face crossed
by another axis) is read from the same array slot by both — no separate
reconciliation step is needed.

`split_segment` is available for two dimensions and up, matching
[`HermiteGrid2D`–`4D`](hermite_grid.md).
