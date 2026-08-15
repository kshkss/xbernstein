r"""Direct tensor-product Bernstein Hermite interpolation on $[0,1]^d$.

For parameter order $\mathbf{u}=(u_0,\ldots,u_{d-1})$, every input vertex is
indexed by $\mathbf{v}\in\{0,1\}^d$ in that same order.  The value argument
``f`` stores $f_{\mathbf{v}}=p(\mathbf{v})$ on its final $d$ axes:
$f[...,v_0,\ldots,v_{d-1}]$.  Derivative arguments are grouped by total
order $s=|\alpha|$, where
$\alpha=(\alpha_0,\ldots,\alpha_{d-1})$ denotes
$\partial^\alpha p=\partial_{u_0}^{\alpha_0}\cdots
\partial_{u_{d-1}}^{\alpha_{d-1}}p$.  Their final $d$ vertex axes retain
the same $(u_0,\ldots,u_{d-1})$ order; a final derivative-kind axis follows
them when a group contains multiple $\alpha$ values.

Within a total-order group, multi-indices use descending lexicographic order.
For example, 2D cubic data is ``(f, d1, d2)`` with
$d_1[...,v_x,v_y,0]=p_x(v_x,v_y)$,
$d_1[...,v_x,v_y,1]=p_y(v_x,v_y)$, and
$d_2[...,v_x,v_y]=p_{xy}(v_x,v_y)$.  Singleton groups omit the final
derivative-kind axis.

For degree $n=2r-1$, direct endpoint conversion uses

$$
c_i=\sum_{k=0}^{i}\binom{i}{k}\frac{(n-k)!}{n!}p^{(k)}(0),
\qquad
c_{n-i}=\sum_{k=0}^{i}(-1)^k\binom{i}{k}
\frac{(n-k)!}{n!}p^{(k)}(1),
\quad 0\leq i<r.
$$

Applying this identity tensor-product-wise yields controls
$c_{\mathbf{i}}$ directly from vertex derivatives.  The returned coefficient
array has shape ``(*batch, n + 1, ..., n + 1)``: its final $d$ Bernstein axes
are ordered by $(u_0,\ldots,u_{d-1})$, exactly like the input vertex axes.
No power-basis coefficients are formed.
"""

import itertools
import math

from beartype import beartype
import jax
import jax.numpy as jnp
from jaxtyping import Float, Shaped, jaxtyped

from .bernstein import Bernstein
from .bernstein_2d import Bernstein2D
from .bernstein_3d import Bernstein3D
from .bernstein_4d import Bernstein4D


def _multi_indices(dimensions: int, limit: int, total: int) -> tuple[tuple[int, ...], ...]:
    return sorted(
        (
            alpha
            for alpha in itertools.product(range(limit), repeat=dimensions)
            if sum(alpha) == total
        ),
        reverse=True,
    )


def _validate_groups(
    dimensions: int,
    degree: int,
    groups: tuple[Float[jax.Array, "..."], ...],
) -> tuple[tuple[jax.Array, ...], tuple[int, ...]]:
    r = (degree + 1) // 2
    expected_groups = (r - 1) * dimensions + 1
    if len(groups) != expected_groups:
        raise TypeError(
            f"degree-{degree} {dimensions}D interpolation requires "
            f"{expected_groups} derivative groups"
        )
    values = tuple(jnp.asarray(group) for group in groups)
    batch_shape = values[0].shape[:-dimensions]
    vertex_shape = (2,) * dimensions
    if values[0].shape != batch_shape + vertex_shape:
        raise ValueError(
            "the value group must end with one length-2 axis per dimension"
        )
    for total, group in enumerate(values[1:], start=1):
        count = len(_multi_indices(dimensions, r, total))
        expected_shape = batch_shape + vertex_shape + (() if count == 1 else (count,))
        if group.shape != expected_shape:
            raise ValueError(
                f"derivative group {total} must have shape {expected_shape}"
            )
    return values, batch_shape


def _interpolate(
    dimensions: int,
    degree: int,
    groups: tuple[Float[jax.Array, "..."], ...],
) -> Bernstein | Bernstein2D | Bernstein3D | Bernstein4D:
    r = (degree + 1) // 2
    expected_groups = (r - 1) * dimensions + 1
    values, batch_shape = _validate_groups(dimensions, degree, groups)

    coefficients = jnp.zeros(batch_shape + (degree + 1,) * dimensions, values[0].dtype)
    scales = tuple(
        math.factorial(degree - k) / math.factorial(degree) for k in range(r)
    )
    for control_index in itertools.product(range(degree + 1), repeat=dimensions):
        vertex = tuple(0 if index < r else 1 for index in control_index)
        coefficient = jnp.zeros(batch_shape, values[0].dtype)
        for total in range(expected_groups):
            for derivative_index, alpha in enumerate(
                _multi_indices(dimensions, r, total)
            ):
                weight = 1.0
                for index, derivative_order, endpoint in zip(
                    control_index, alpha, vertex
                ):
                    distance = index if endpoint == 0 else degree - index
                    if derivative_order > distance:
                        weight = 0.0
                        break
                    sign = 1.0 if endpoint == 0 else (-1.0) ** derivative_order
                    weight *= (
                        sign
                        * math.comb(distance, derivative_order)
                        * scales[derivative_order]
                    )
                if total == 0:
                    derivative = values[0][(...,) + vertex]
                elif len(_multi_indices(dimensions, r, total)) == 1:
                    derivative = values[total][(...,) + vertex]
                else:
                    derivative = values[total][(...,) + vertex + (derivative_index,)]
                coefficient = coefficient + weight * derivative
        coefficients = coefficients.at[(...,) + control_index].set(coefficient)

    polynomial_types = (None, Bernstein, Bernstein2D, Bernstein3D, Bernstein4D)
    return polynomial_types[dimensions](coefficients)


@jaxtyped(typechecker=beartype)
def linear_interpolate_1d(
    f: Float[jax.Array, "*batch 2"],
) -> Shaped[Bernstein, "*batch"]:
    r"""Interpolate two endpoint values by the unique degree-1 Bernstein polynomial.

    The result is the affine function

    $$
    p(x)=\sum_{i=0}^{1}c_i B_i^1(x)
    $$

    satisfying

    $$
    p(v)=f(v),\qquad v\in\{0,1\}.
    $$

    Thus all supplied data, namely the two endpoint values, are reproduced
    exactly. ``f`` has shape ``(*batch,2)``:
    ``f[...,0]=f(0)`` and ``f[...,1]=f(1)``. The result has
    ``bpoly.order == 1`` and ``bpoly.c[...,i_x]`` follows the same $x$ order
    as the vertex axis. Leading batch axes are preserved.
    """
    return _interpolate(1, 1, (f,))


@jaxtyped(typechecker=beartype)
def linear_interpolate_2d(
    f: Float[jax.Array, "*batch 2 2"],
) -> Shaped[Bernstein2D, "*batch"]:
    r"""Interpolate vertex values by the unique bilinear Bernstein polynomial.

    The result is

    $$
    p(x,y)=\sum_{i=0}^{1}\sum_{j=0}^{1}
    c_{ij}B_i^1(x)B_j^1(y),
    $$

    the unique polynomial of degree at most one in each coordinate satisfying

    $$
    p(v_x,v_y)=f(v_x,v_y),
    \qquad (v_x,v_y)\in\{0,1\}^2.
    $$

    Hence all four supplied vertex values are reproduced exactly. ``f`` has
    shape ``(*batch,2,2)`` and
    ``f[...,v_x,v_y]=f(v_x,v_y)``. The result has
    ``bpoly.order == [1,1]``; ``bpoly.c[...,i_x,i_y]`` uses the same $x,y$
    axis order. Leading batch axes are preserved.
    """
    return _interpolate(2, 1, (f,))


@jaxtyped(typechecker=beartype)
def linear_interpolate_3d(
    f: Float[jax.Array, "*batch 2 2 2"],
) -> Shaped[Bernstein3D, "*batch"]:
    r"""Interpolate vertex values by the unique trilinear Bernstein polynomial.

    Writing $\mathbf{x}=(x,y,z)$, the result is

    $$
    p(\mathbf{x})=
    \sum_{\mathbf{i}\in\{0,1\}^3}
    c_{\mathbf{i}}\prod_{a=1}^{3}B_{i_a}^1(x_a)
    \in\mathcal Q_1,
    $$

    where $\mathcal Q_1$ contains polynomials of degree at most one in each
    coordinate. It is uniquely determined by

    $$
    p(\mathbf v)=f(\mathbf v),
    \qquad \mathbf v\in\{0,1\}^3.
    $$

    Thus all eight supplied vertex values are reproduced exactly.
    ``f[...,v_x,v_y,v_z]`` has shape ``(*batch,2,2,2)`` and stores
    $f(v_x,v_y,v_z)$. The result has ``bpoly.order == [1,1,1]``;
    ``bpoly.c[...,i_x,i_y,i_z]`` preserves the $x,y,z$ order and all leading
    batch axes.
    """
    return _interpolate(3, 1, (f,))


@jaxtyped(typechecker=beartype)
def linear_interpolate_4d(
    f: Float[jax.Array, "*batch 2 2 2 2"],
) -> Shaped[Bernstein4D, "*batch"]:
    r"""Interpolate vertex values by the unique 4D multilinear Bernstein polynomial.

    For $\mathbf{x}=(x,y,z,w)$, the result is

    $$
    p(\mathbf{x})=
    \sum_{\mathbf{i}\in\{0,1\}^4}
    c_{\mathbf{i}}\prod_{a=1}^{4}B_{i_a}^1(x_a)
    \in\mathcal Q_1,
    $$

    where the degree is at most one in each coordinate, not total degree one.
    The $16$ conditions

    $$
    p(\mathbf v)=f(\mathbf v),
    \qquad \mathbf v\in\{0,1\}^4,
    $$

    uniquely determine $p$ and reproduce every supplied vertex value.
    ``f[...,v_x,v_y,v_z,v_w]`` has shape ``(*batch,2,2,2,2)``. The result
    has ``bpoly.order == [1,1,1,1]``;
    ``bpoly.c[...,i_x,i_y,i_z,i_w]`` preserves the $x,y,z,w$ order and all
    leading batch axes.
    """
    return _interpolate(4, 1, (f,))


@jaxtyped(typechecker=beartype)
def hermite_interpolate_1d(
    f: Float[jax.Array, "*batch 2"], d1: Float[jax.Array, "*batch 2"]
) -> Shaped[Bernstein, "*batch"]:
    r"""Interpolate endpoint values and slopes by the unique cubic polynomial.

    The result

    $$
    p(x)=\sum_{i=0}^{3}c_iB_i^3(x)
    $$

    is uniquely determined by the Hermite conditions

    $$
    p^{(k)}(v)=f^{(k)}(v),
    \qquad v\in\{0,1\},\quad k\in\{0,1\}.
    $$

    Hence both the value and first derivative are reproduced at both
    endpoints. ``f[...,v_x]`` and ``d1[...,v_x]`` have shape
    ``(*batch,2)`` and store $f(v_x)$ and $f_x(v_x)$. The result has
    ``bpoly.order == 3``; ``bpoly.c[...,i_x]`` is in $x$ order and leading
    batch axes are preserved.
    """
    return _interpolate(1, 3, (f, d1))


@jaxtyped(typechecker=beartype)
def hermite_interpolate_2d(
    f: Float[jax.Array, "*batch 2 2"],
    d1: Float[jax.Array, "*batch 2 2 2"],
    d2: Float[jax.Array, "*batch 2 2"],
) -> Shaped[Bernstein2D, "*batch"]:
    r"""Interpolate the first-order vertex jet by a bicubic Bernstein polynomial.

    The result is the unique tensor-product polynomial

    $$
    p(x,y)=\sum_{i=0}^{3}\sum_{j=0}^{3}
    c_{ij}B_i^3(x)B_j^3(y)\in\mathcal Q_3
    $$

    satisfying

    $$
    \partial^\alpha p(\mathbf v)=\partial^\alpha f(\mathbf v),
    \qquad
    \mathbf v\in\{0,1\}^2,\quad
    \alpha\in\{0,1\}^2.
    $$

    Thus $f$, $f_x$, $f_y$, and $f_{xy}$ are reproduced at every vertex.
    Degree $(3,3)$ means degree at most three in each coordinate.

    **Inputs:** vertex axes are ``v_x,v_y``:
    ``f[...,v_x,v_y]=f(v_x,v_y)``.
    ``d1[...,v_x,v_y,0/1]=(f_x,f_y)`` and
    ``d2[...,v_x,v_y]=f_xy``.
    **Returns:** ``bpoly.order == [3,3]``;
    ``bpoly.c[...,i_x,i_y]`` is in the same $x,y$ order as the vertex axes.
    Leading batch axes are preserved.
    """
    return _interpolate(2, 3, (f, d1, d2))


def hermite_interpolate_3d(
    f: Float[jax.Array, "*batch 2 2 2"],
    d1: Float[jax.Array, "*batch 2 2 2 3"],
    d2: Float[jax.Array, "*batch 2 2 2 3"],
    d3: Float[jax.Array, "*batch 2 2 2"],
) -> Shaped[Bernstein3D, "*batch"]:
    r"""Interpolate the first-order vertex jet by a tricubic Bernstein polynomial.

    For $\mathbf{x}=(x,y,z)$, the result is the unique
    $p\in\mathcal Q_3$,

    $$
    p(\mathbf{x})=
    \sum_{\mathbf i\in\{0,\ldots,3\}^3}
    c_{\mathbf i}\prod_{a=1}^{3}B_{i_a}^3(x_a),
    $$

    such that

    $$
    \partial^\alpha p(\mathbf v)=\partial^\alpha f(\mathbf v),
    \qquad
    \mathbf v\in\{0,1\}^3,\quad
    \alpha\in\{0,1\}^3.
    $$

    It therefore reproduces at every vertex the value; $f_x,f_y,f_z$;
    $f_{xy},f_{xz},f_{yz}$; and $f_{xyz}$. Degree $(3,3,3)$ is
    coordinate-wise, not total degree.

    ``f[...,v_x,v_y,v_z]`` and ``d3[...,v_x,v_y,v_z]=f_xyz`` have shape
    ``(*batch,2,2,2)``.
    ``d1[...,v_x,v_y,v_z,0/1/2]=(f_x,f_y,f_z)`` and
    ``d2[...,v_x,v_y,v_z,0/1/2]=(f_xy,f_xz,f_yz)``.
    The result has ``bpoly.order == [3,3,3]``;
    ``bpoly.c[...,i_x,i_y,i_z]`` preserves $x,y,z$ order and leading batch
    axes.
    """
    return _interpolate(3, 3, (f, d1, d2, d3))


def hermite_interpolate_4d(
    f: Float[jax.Array, "*batch 2 2 2 2"],
    d1: Float[jax.Array, "*batch 2 2 2 2 4"],
    d2: Float[jax.Array, "*batch 2 2 2 2 6"],
    d3: Float[jax.Array, "*batch 2 2 2 2 4"],
    d4: Float[jax.Array, "*batch 2 2 2 2"],
) -> Shaped[Bernstein4D, "*batch"]:
    r"""Interpolate the first-order vertex jet by a 4D tensor-cubic polynomial.

    For $\mathbf{x}=(x,y,z,w)$, the result is the unique
    $p\in\mathcal Q_3$,

    $$
    p(\mathbf{x})=
    \sum_{\mathbf i\in\{0,\ldots,3\}^4}
    c_{\mathbf i}\prod_{a=1}^{4}B_{i_a}^3(x_a),
    $$

    satisfying

    $$
    \partial^\alpha p(\mathbf v)=\partial^\alpha f(\mathbf v),
    \qquad
    \mathbf v\in\{0,1\}^4,\quad
    \alpha\in\{0,1\}^4.
    $$

    Consequently every derivative using each coordinate zero or one time is
    reproduced at every vertex: the value, four first derivatives, six
    second-order mixed derivatives, four third-order mixed derivatives, and
    $f_{xyzw}$. Degree $(3,3,3,3)$ is coordinate-wise.

    **Inputs:** ``f[...,v_x,v_y,v_z,v_w]=f(v_x,v_y,v_z,v_w)``.
    ``d1[...,v_x,v_y,v_z,v_w,0..3]=(f_x,f_y,f_z,f_w)``,
    ``d2[...,v_x,v_y,v_z,v_w,0..5]=(f_xy,f_xz,f_xw,f_yz,f_yw,f_zw)``,
    ``d3[...,v_x,v_y,v_z,v_w,0..3]=(f_xyz,f_xyw,f_xzw,f_yzw)``, and
    ``d4[...,v_x,v_y,v_z,v_w]=f_xyzw``.
    **Returns:** ``bpoly.order == [3,3,3,3]``;
    ``bpoly.c[...,i_x,i_y,i_z,i_w]`` is in $x,y,z,w$ order. Leading batch
    axes are preserved.
    """
    return _interpolate(4, 3, (f, d1, d2, d3, d4))


def quintic_hermite_interpolate_1d(
    f: Float[jax.Array, "*batch 2"],
    d1: Float[jax.Array, "*batch 2"],
    d2: Float[jax.Array, "*batch 2"],
) -> Shaped[Bernstein, "*batch"]:
    r"""Interpolate endpoint values through second derivatives by a quintic.

    The result

    $$
    p(x)=\sum_{i=0}^{5}c_iB_i^5(x)
    $$

    is the unique degree-at-most-five polynomial satisfying

    $$
    p^{(k)}(v)=f^{(k)}(v),
    \qquad v\in\{0,1\},\quad k\in\{0,1,2\}.
    $$

    Thus the supplied value, first derivative, and second derivative are all
    reproduced at both endpoints.

    **Inputs:** ``f[...,v_x]=f(v_x)``, ``d1[...,v_x]=f_x(v_x)``, and
    ``d2[...,v_x]=f_xx(v_x)``; all have shape ``(*batch,2)``.
    **Returns:** ``bpoly.order == 5``; ``bpoly.c[...,i_x]`` is in $x$ order.
    Leading batch axes are preserved.
    """
    return _interpolate(1, 5, (f, d1, d2))


def quintic_hermite_interpolate_2d(
    f: Float[jax.Array, "*batch 2 2"],
    d1: Float[jax.Array, "*batch 2 2 2"],
    d2: Float[jax.Array, "*batch 2 2 3"],
    d3: Float[jax.Array, "*batch 2 2 2"],
    d4: Float[jax.Array, "*batch 2 2"],
) -> Shaped[Bernstein2D, "*batch"]:
    r"""Interpolate the second-order vertex jet by a biquintic polynomial.

    The result is the unique tensor-product polynomial

    $$
    p(x,y)=\sum_{i=0}^{5}\sum_{j=0}^{5}
    c_{ij}B_i^5(x)B_j^5(y)\in\mathcal Q_5
    $$

    satisfying

    $$
    \partial^\alpha p(\mathbf v)=\partial^\alpha f(\mathbf v),
    \qquad
    \mathbf v\in\{0,1\}^2,\quad
    \alpha\in\{0,1,2\}^2.
    $$

    Hence all nine derivatives formed by differentiating zero, one, or two
    times in each coordinate are reproduced at every vertex:
    $f,f_x,f_y,f_{xx},f_{xy},f_{yy},f_{xxy},f_{xyy},f_{xxyy}$.
    Degree $(5,5)$ is coordinate-wise, not total degree.

    **Inputs:** vertex axes are ``v_x,v_y``:
    ``f[...,v_x,v_y]=f(v_x,v_y)``.
    ``d1[...,v_x,v_y,0/1]=(f_x,f_y)``,
    ``d2[...,v_x,v_y,0/1/2]=(f_xx,f_xy,f_yy)``,
    ``d3[...,v_x,v_y,0/1]=(f_xxy,f_xyy)``, and
    ``d4[...,v_x,v_y]=f_xxyy``.
    **Returns:** ``bpoly.order == [5,5]``;
    ``bpoly.c[...,i_x,i_y]`` is in the same $x,y$ order as the vertex axes.
    Leading batch axes are preserved.
    """
    return _interpolate(2, 5, (f, d1, d2, d3, d4))


def quintic_hermite_interpolate_3d(
    f: Float[jax.Array, "*batch 2 2 2"],
    d1: Float[jax.Array, "*batch 2 2 2 3"],
    d2: Float[jax.Array, "*batch 2 2 2 6"],
    d3: Float[jax.Array, "*batch 2 2 2 7"],
    d4: Float[jax.Array, "*batch 2 2 2 6"],
    d5: Float[jax.Array, "*batch 2 2 2 3"],
    d6: Float[jax.Array, "*batch 2 2 2"],
) -> Shaped[Bernstein3D, "*batch"]:
    r"""Interpolate the second-order vertex jet by a triquintic polynomial.

    For $\mathbf{x}=(x,y,z)$, the result is the unique
    $p\in\mathcal Q_5$,

    $$
    p(\mathbf{x})=
    \sum_{\mathbf i\in\{0,\ldots,5\}^3}
    c_{\mathbf i}\prod_{a=1}^{3}B_{i_a}^5(x_a),
    $$

    satisfying

    $$
    \partial^\alpha p(\mathbf v)=\partial^\alpha f(\mathbf v),
    \qquad
    \mathbf v\in\{0,1\}^3,\quad
    \alpha\in\{0,1,2\}^3.
    $$

    Therefore all $27$ supplied derivative kinds, including every mixed
    derivative through $f_{xxyyzz}$, are reproduced at all eight vertices.
    Degree $(5,5,5)$ means degree at most five in each coordinate.

    **Inputs:** vertex axes are ``v_x,v_y,v_z``:
    ``f[...,v_x,v_y,v_z]=f(v_x,v_y,v_z)``.
    ``d1[...,v_x,v_y,v_z,0/1/2]=(f_x,f_y,f_z)``,
    ``d2[...,v_x,v_y,v_z,0/1/2/3/4/...]=(f_xx,f_xy,f_xz,f_yy,f_yz,...)``,
    ``d3[...,v_x,v_y,v_z,0/1/2/3/4/...]=(f_xxy,f_xxz,f_xyy,f_xyz,f_xzz,...)``,
    ``d4[...,v_x,v_y,v_z,0/1/2/3/4/...]=(f_xxyy,f_xxyz,f_xxzz,f_xyyz,f_xyzz,...)``,
    ``d5[...,v_x,v_y,v_z,0/1/2]=(f_xxyyz,f_xxyzz,f_xyyzz)``, and
    ``d6[...,v_x,v_y,v_z]=f_xxyyzz``.
    **Returns:** ``bpoly.order == [5,5,5]``;
    ``bpoly.c[...,i_x,i_y,i_z]`` is in the same $x,y,z$ order as the vertex
    axes. Leading batch axes are preserved.
    """
    return _interpolate(3, 5, (f, d1, d2, d3, d4, d5, d6))


def quintic_hermite_interpolate_4d(
    f: Float[jax.Array, "*batch 2 2 2 2"],
    d1: Float[jax.Array, "*batch 2 2 2 2 4"],
    d2: Float[jax.Array, "*batch 2 2 2 2 10"],
    d3: Float[jax.Array, "*batch 2 2 2 2 16"],
    d4: Float[jax.Array, "*batch 2 2 2 2 19"],
    d5: Float[jax.Array, "*batch 2 2 2 2 16"],
    d6: Float[jax.Array, "*batch 2 2 2 2 10"],
    d7: Float[jax.Array, "*batch 2 2 2 2 4"],
    d8: Float[jax.Array, "*batch 2 2 2 2"],
) -> Shaped[Bernstein4D, "*batch"]:
    r"""Interpolate the second-order vertex jet by a 4D tensor-quintic polynomial.

    For $\mathbf{x}=(x,y,z,w)$, the result is the unique
    $p\in\mathcal Q_5$,

    $$
    p(\mathbf{x})=
    \sum_{\mathbf i\in\{0,\ldots,5\}^4}
    c_{\mathbf i}\prod_{a=1}^{4}B_{i_a}^5(x_a),
    $$

    satisfying

    $$
    \partial^\alpha p(\mathbf v)=\partial^\alpha f(\mathbf v),
    \qquad
    \mathbf v\in\{0,1\}^4,\quad
    \alpha\in\{0,1,2\}^4.
    $$

    Thus all $81$ supplied derivative kinds, from the value through
    $f_{xxyyzzww}$, are reproduced at all $16$ vertices. Degree
    $(5,5,5,5)$ is coordinate-wise rather than total degree.

    **Inputs:** vertex axes are ``v_x,v_y,v_z,v_w``:
    ``f[...,v_x,v_y,v_z,v_w]=f(v_x,v_y,v_z,v_w)``.
    ``d1[...,v_x,v_y,v_z,v_w,0/1/2/3]=(f_x,f_y,f_z,f_w)``,
    ``d2[...,v_x,v_y,v_z,v_w,0/1/2/3/4/...]=(f_xx,f_xy,f_xz,f_xw,f_yy,...)``,
    ``d3[...,v_x,v_y,v_z,v_w,0/1/2/3/4/...]=(f_xxy,f_xxz,f_xxw,f_xyy,f_xyz,...)``,
    ``d4[...,v_x,v_y,v_z,v_w,0/1/2/3/4/...]=(f_xxyy,f_xxyz,f_xxyw,f_xxzz,f_xxzw,...)``,
    ``d5[...,v_x,v_y,v_z,v_w,0/1/2/3/4/...]=(f_xxyyz,f_xxyyw,f_xxyzz,f_xxyzw,f_xxyww,...)``,
    ``d6[...,v_x,v_y,v_z,v_w,0/1/2/3/4/...]=(f_xxyyzz,f_xxyyzw,f_xxyyww,f_xxyzzw,f_xxyzww,...)``,
    ``d7[...,v_x,v_y,v_z,v_w,0/1/2/3]=(f_xxyyzzw,f_xxyyzww,f_xxyzzww,f_xyyzzww)``,
    and ``d8[...,v_x,v_y,v_z,v_w]=f_xxyyzzww``.
    **Returns:** ``bpoly.order == [5,5,5,5]``;
    ``bpoly.c[...,i_x,i_y,i_z,i_w]`` uses the same $x,y,z,w$ order as the
    vertex axes. Leading batch axes are preserved.
    """
    return _interpolate(4, 5, (f, d1, d2, d3, d4, d5, d6, d7, d8))
