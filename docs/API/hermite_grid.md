# Hermite grid interpolation

All grid classes convert the physical vertex jets supplied to their
constructors into cell-local Bernstein coefficients. The original vertex
arrays are not retained. For a grid with leading batch shape `batch_shape` and
cell counts `cell_shape`, `coefficients` has shape
`batch_shape + cell_shape + (degree + 1,) * dimension`.

The `f` and `d1`, ... derivative properties reconstruct physical vertex data
from these coefficients when accessed. Derivatives remain grouped by total
order in descending lexicographic multi-index order. At an interior vertex the
cell on its positive side is used; at an upper domain boundary the final cell
is used. Because reconstruction differentiates floating-point Bernstein
coefficients, its result can differ from the constructor input by rounding
error, especially for high-order mixed derivatives.

## Cubic grids

::: xbernstein.HermiteGrid1D

::: xbernstein.HermiteGrid2D

::: xbernstein.HermiteGrid3D

::: xbernstein.HermiteGrid4D

## Quintic grids

The quintic classes accept the complete vertex jet with each coordinate
differentiated at most twice. Derivative kinds use descending lexicographic
multi-index order, matching the corresponding
`quintic_hermite_interpolate_*d` function.

::: xbernstein.QuinticHermiteGrid1D

::: xbernstein.QuinticHermiteGrid2D

::: xbernstein.QuinticHermiteGrid3D

::: xbernstein.QuinticHermiteGrid4D

## Segment decomposition

::: xbernstein.GridSegment
