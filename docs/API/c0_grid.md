# C0 tensor-product Bernstein grids

## One-dimensional

::: xbernstein.P1C0Grid1D

::: xbernstein.P2C0Grid1D

::: xbernstein.P3C0Grid1D

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
