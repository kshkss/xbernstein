# Curves and rectilinear grids

`PiecewiseRationalCurve3D` stores node positions, unit tangent directions,
and curvature vectors. Each interval is represented by a cubic rational
Bernstein curve, and adjacent intervals share the prescribed $G^2$ endpoint
geometry.

`RationalHermiteGrid3D` stores scalar vertex jets on a nonuniform rectilinear
grid. Each cell is mapped to local coordinates in $[0,1]^3$ and interpolated
by a tricubic rational tensor-product Bernstein function.
