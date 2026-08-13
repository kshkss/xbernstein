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
- **Rectilinear grid interpolation** — evaluate and traverse nonuniform 3D
  grids of tricubic rational Hermite patches.
- **Piecewise spatial curves** — construct G² cubic rational Bernstein
  segments from 3D node positions, tangent directions, and curvatures.
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
| 3D rational Hermite grid | `RationalHermiteGrid3D`, `GridSegment3D` |
| Piecewise 3D G² curve | `PiecewiseRationalCurve3D` |
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
