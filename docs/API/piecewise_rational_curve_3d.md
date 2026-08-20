# Piecewise rational 3D curves

## PiecewiseRationalCurve3D

::: xbernstein.PiecewiseRationalCurve3D

The constructor converts every segment once to cubic homogeneous Bernstein
coordinates and stores endpoint Hermite data for ``(X, Y, Z, W)``. Euclidean
positions are evaluated as ``(X/W, Y/W, Z/W)``. The ``positions``,
``tangents``, and ``curvatures`` attributes are properties reconstructed from
the homogeneous data when accessed.

The first segment retains its original parameterization and has endpoint
weights equal to one; its two interior weights remain those produced by the
G2 solver. Subsequent segments are reparameterized by endpoint-preserving
Mobius maps. Their scale factors are propagated from left to right so that
Euclidean velocity is C1 at integer knots and the homogeneous weight value is
shared by adjacent segments. The final endpoint weight is not normalized back
to one.
