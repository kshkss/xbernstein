# Piecewise quintic 3D curves

## PiecewiseQuinticCurve3D

::: xbernstein.PiecewiseQuinticCurve3D

Each segment's curvature vector uses the classical, speed-independent
convention shared with [`PiecewiseRationalCurve3D`](piecewise_rational_curve_3d.md):
the component of acceleration normal to velocity, divided by the squared
speed. Torsion is the standard parameterization-independent geometric torsion
`(v x a) . jerk / |v x a|^2`.

The constructor solves each segment's two free Bezier controls (its
`interior_coefficients` under [`P5C1`](fem1d.md)) directly from the endpoint
curvature vectors and torsion values, using the fact that the local binormal
direction `v x a` does not depend on the (otherwise unconstrained) tangential
acceleration. Because those two free controls are independent per segment,
only position and velocity are guaranteed continuous across a shared node:
curvature and torsion may jump between two segments meeting at the same
point, matching `P5C1`'s own C1-only continuity.

`append` returns a new curve with one more point and one more segment; it
never mutates the original instance.
