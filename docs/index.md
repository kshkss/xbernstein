---
icon: lucide/rocket
---


# Get started

`xbernstein` is JAX-compatible Bernstein polynomial operations for tensor-product, simplex,
and rational functions.

`xbernstein` provides immutable polynomial objects that work with JAX arrays
and transformations. It is designed for numerical geometry and optimization
workflows that need Bernstein-basis operations without leaving JAX.

## Features

- **JAX-native computation** — use JIT compilation, vectorization, and batched
  coefficient arrays.
- **Multiple domains** — work on `[0, 1]`, tensor-product domains
  `[0, 1]^d`, or simplices in two to four dimensions.
- **Polynomial and rational forms** — represent ordinary Bernstein
  polynomials and positive-weight rational Bernstein functions.
- **Basis-preserving operations** — evaluate, differentiate, integrate, split,
  slice, restrict to segments, and combine polynomials.
- **Interpolation helpers** — construct linear, Hermite, quintic Hermite, and
  rational Hermite interpolants in one to four dimensions.
- **Rectilinear grid interpolation** — evaluate nonuniform 1D–4D grids of
  cubic Hermite patches and traverse 2D–4D grids by line segments.
- **Finite-element integration** — integrate Q1/Q2 Bernstein elements,
  including embedded and rational geometries, and form local FEM matrices.
- **1D finite-element interpolation** — evaluate piecewise P1–P5 fields with
  C0, C1, or C2 continuity across cells, from explicit node, connectivity,
  and degree-of-freedom data.
- **Piecewise spatial curves** — construct G² cubic rational Bernstein
  segments from 3D node positions, tangent directions, and curvatures.
- **Piecewise cubic Hermite curves** — construct C¹ continuous vector-valued
  Bernstein segments from 3D positions and tangent vectors.
- **Global optimization** — approximate minima and maxima with
  Bernstein-basis branch-and-bound solvers.

## Installation

`xbernstein` requires Python 3.11 or later. Install the latest version directly
from GitHub:

```console
python -m pip install "xbernstein @ git+https://github.com/kshkss/xbernstein.git"
```

The package installs JAX as a dependency. For GPU, TPU, or other
platform-specific JAX builds, follow the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Quick start

Create a polynomial from its Bernstein coefficients, evaluate it, and apply
operations that return new Bernstein objects:

```python
import jax
import jax.numpy as jnp

from xbernstein import Bernstein, minimize

polynomial = Bernstein(jnp.array([1.0, -1.0, 2.0]))

points = jnp.linspace(0.0, 1.0, 5)
evaluate = jax.jit(lambda x: polynomial(x))
values = evaluate(points)

derivative = polynomial.deriv()
left, right = polynomial.split(jnp.array(0.5))
minimum = minimize(polynomial)

print(values)
print(minimum.f, minimum.x)
```

The final coefficient axis stores the Bernstein coefficients. Any leading
axes are preserved as batch or value axes, so the same operations can process
many polynomials at once.

## Finite-element integration

The finite-element API performs local integration on one mapped element. A
typical workflow is:

1. describe the physical geometry with a vector-valued Bernstein map,
2. choose a scalar Q1 or Q2 reference space,
3. pair the space and geometry as a mapped element, and
4. integrate a callback or use one of the standard local-form helpers.

The following Q1 example maps the reference square $(u,v)\in[0,1]^2$ to the
physical rectangle $(x,y)=(2u,3v)$, then forms the quantities commonly needed
for a scalar Poisson problem:

```python
import jax.numpy as jnp

from xbernstein import (
    Bernstein2D,
    C0Element,
    ElementGeometry,
    Facet,
    MappedElement,
    facet_load_vector,
    gauss_legendre,
    integrate_element,
    load_vector,
    mass_matrix,
    stiffness_matrix,
)

# The leading coefficient axis contains the physical components x and y.
mapping = Bernstein2D(
    jnp.array(
        [
            [[0.0, 0.0], [2.0, 2.0]],  # x(u, v) = 2u
            [[0.0, 3.0], [0.0, 3.0]],  # y(u, v) = 3v
        ]
    )
)

space = C0Element(dimensions=2, degree=1)
element = MappedElement(space, ElementGeometry(mapping))
quadrature = gauss_legendre(dimensions=2, points_per_axis=2)

area = integrate_element(element, lambda point: 1.0, quadrature)
mass = mass_matrix(element)
stiffness = stiffness_matrix(element)

# A position-dependent volume source f(x, y) = x + y.
volume_load = load_vector(
    element,
    lambda point: jnp.sum(point.physical_coordinates),
)

# A constant Neumann source on u = 1, the right edge of this mapping.
right_facet = Facet(axis=0, side=1)
boundary_load = facet_load_vector(element, right_facet, source=1.0)

print(float(area))
print(mass.shape, stiffness.shape)
print(volume_load.shape, boundary_load.shape)
```

This prints:

```text
6.0
(4, 4) (4, 4)
(4,) (4,)
```

An integration callback receives an `ElementPoint` (or a `FacetPoint` for a
facet integral). The integration function applies both the quadrature weight
and the physical Jacobian measure, so the callback must not multiply either
factor itself. The standard helpers choose a default quadrature rule; pass an
explicit rule when a curved or rational geometry needs a convergence study.

These operations return element-local arrays. Mesh topology, global degree-of-
freedom numbering, orientation matching between neighboring elements, sparse
assembly, and boundary-condition elimination are intentionally left to the
calling application. See the
[finite-element API reference](API/finite_element.md) for coefficient layouts,
local freedom ordering, callback fields, facet conventions, and quadrature
guidance.

## API overview

| Domain and representation | Public API |
| --- | --- |
| One-dimensional polynomial | `Bernstein` |
| Tensor-product polynomial | `Bernstein2D`, `Bernstein3D`, `Bernstein4D` |
| Simplex polynomial | `Bernstein2DS`, `Bernstein3DS`, `Bernstein4DS` |
| One-dimensional rational function | `RationalBernstein` |
| Tensor-product rational function | `RationalBernstein2D`, `RationalBernstein3D`, `RationalBernstein4D` |
| Rational simplex function | `RationalBernstein2DS`, `RationalBernstein3DS`, `RationalBernstein4DS` |
| Interpolation | `linear_interpolate_*d`, `hermite_interpolate_*d`, `quintic_hermite_interpolate_*d`, `rational_hermite_interpolate_*d` |
| Hermite grid interpolation | `HermiteGrid1D`–`4D`, `QuinticHermiteGrid1D`–`4D`, `GridSegment` |
| C0 finite-element grid interpolation | `P1C0Grid1D`–`4D`, `P2C0Grid1D`–`4D`, `P3C0Grid1D`–`4D` |
| C1 finite-element grid interpolation | `P3C1Grid1D`–`4D`, `P4C1Grid1D`–`4D`, `P5C1Grid1D`–`4D` |
| Finite-element integration | `C0Element`, `ElementGeometry`, `MappedElement`, `integrate_element`, `integrate_facet` |
| 1D finite-element interpolation | `P1C0`, `P2C0`, `P3C0`, `P3C1`, `P4C1`, `P5C1`, `P5C2` |
| 3D rational Hermite grid | `RationalHermiteGrid3D`, `GridSegment3D` |
| Piecewise 3D G² curve | `PiecewiseRationalCurve3D` |
| Piecewise 3D C¹ curve | `PiecewiseCurve3D` |
| Piecewise 3D quintic curve (curvature/torsion) | `PiecewiseQuinticCurve3D` |
| Optimization | `minimize`, `maximize` |

Tensor-product classes use one trailing coefficient axis per parameter.
Simplex classes pack coefficients into one trailing axis and evaluate from
barycentric coordinates. Rational classes take dehomogenized control values
and strictly positive weights:

```python
import jax.numpy as jnp

from xbernstein import RationalBernstein2D

surface = RationalBernstein2D(
    values=jnp.array([[0.0, 1.0], [1.0, 2.0]]),
    weights=jnp.array([[1.0, 2.0], [1.0, 1.0]]),
)

value = surface(0.25, 0.75)
diagonal = surface.segment(jnp.zeros(2), jnp.ones(2))
```

See the public exports in
[`src/xbernstein/__init__.py`](src/xbernstein/__init__.py) and the
implementation docstrings for detailed signatures and representation rules.

## License

This project is licensed under the terms in [LICENSE](LICENSE).
