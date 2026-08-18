# Piecewise rational 3D curves

## PiecewiseRationalCurve3D

::: xbernstein.PiecewiseRationalCurve3D

The constructor converts every segment once to cubic homogeneous Bernstein
coordinates and stores endpoint Hermite data for ``(X, Y, Z, W)``. Euclidean
positions are evaluated as ``(X/W, Y/W, Z/W)``. The ``positions``,
``tangents``, and ``curvatures`` attributes are properties reconstructed from
the homogeneous data when accessed.
