# C0 tensor-product Bernstein grids

## One-dimensional

::: xbernstein.P1C0Grid1D

::: xbernstein.P2C0Grid1D

::: xbernstein.P3C0Grid1D

::: xbernstein.C0Grid1D

## Two-dimensional

::: xbernstein.P1C0Grid2D

::: xbernstein.P2C0Grid2D

::: xbernstein.P3C0Grid2D

## Three-dimensional

::: xbernstein.P1C0Grid3D

::: xbernstein.P2C0Grid3D

::: xbernstein.P3C0Grid3D

## Four-dimensional

::: xbernstein.P1C0Grid4D

::: xbernstein.P2C0Grid4D

::: xbernstein.P3C0Grid4D

Each class stores one dense control-point array on a *refined* structured
grid: along every axis, each original grid cell is subdivided into `degree`
equal parameter steps, so a cell's local `(degree + 1) ** dimension`
tensor-product Bernstein control points are simply an overlapping window of
that array (stride `degree`, window size `degree + 1` per axis). Two
neighboring cells therefore share their entire common boundary control-point
sub-array exactly — not just the corner values — which is what makes the
family C0 (not just continuous at grid vertices) for `degree > 1`.

Only entries landing on an original grid vertex (index `0` or `degree` along
every axis) equal the true function value there; every other stored entry is
a raw shared Bernstein coefficient, matching the "freedoms are Bernstein
coefficients, not Lagrange values" convention used elsewhere in this
library. For `degree = 1` the refined grid coincides with the original grid,
so `f` is simply the function's value at every grid vertex (ordinary
multilinear interpolation).

`split_segment` (available for two dimensions and up, matching
[`HermiteGrid2D`–`4D`](hermite_grid.md)) decomposes a straight segment into
its per-cell pieces at grid-line crossings.

`segment_grid` (also available for two dimensions and up) restricts the
interpolant to a straight segment and returns that restriction as an exact
`C0Grid1D`: one whose degree is `dimension * degree`, since each piece is
the tensor cell polynomial's exact 1D restriction along the segment. Its `x`
axis is the Euclidean distance travelled from `start`. Because the number of
pieces depends on where the segment crosses grid lines, `segment_grid`
requires concrete `start`/`end` values and cannot be traced under `jax.jit`.

`integrate_out` (also available for two dimensions and up) integrates the
represented function over one axis' full physical range and returns the
`dimension - 1`-dimensional grid of the same `degree` over the remaining
axes. Because the tensor-product Bernstein basis factorizes across axes,
this reduces to a per-cell, width-weighted sum directly on the shared
refined array — no reconstruction of individual cells is needed. Unlike
`segment_grid`, the result's shape depends only on the grid's static shape,
so `integrate_out` is `jax.jit`/`jax.vmap`-traceable.
