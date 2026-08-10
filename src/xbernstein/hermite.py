r"""Direct tensor-product Bernstein Hermite interpolation on $[0,1]^d$.

For parameter order $\mathbf{u}=(u_0,\ldots,u_{d-1})$, every input vertex is
indexed by $\mathbf{v}\in\{0,1\}^d$ in that same order.  The value argument
``f`` stores $f_{\mathbf{v}}=p(\mathbf{v})$ on its final $d$ axes:
``f[..., v_0, ..., v_{d-1}]``.  Derivative arguments are grouped by total
order $s=|\alpha|$, where
$\alpha=(\alpha_0,\ldots,\alpha_{d-1})$ denotes
$\partial^\alpha p=\partial_{u_0}^{\alpha_0}\cdots
\partial_{u_{d-1}}^{\alpha_{d-1}}p$.  Their final $d$ vertex axes retain
the same $(u_0,\ldots,u_{d-1})$ order; a final derivative-kind axis follows
them when a group contains multiple $\alpha$ values.

Within a total-order group, multi-indices use descending lexicographic order.
For example, 2D cubic data is ``(f, d1, d2)`` with
``d1[..., v_x, v_y, 0] = p_x(v_x, v_y)``,
``d1[..., v_x, v_y, 1] = p_y(v_x, v_y)``, and
``d2[..., v_x, v_y] = p_xy(v_x, v_y)``.  Singleton groups omit the final
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
from jaxtyping import Float, jaxtyped

from .bernstein import Bernstein
from .bernstein_2d import Bernstein2D
from .bernstein_3d import Bernstein3D
from .bernstein_4d import Bernstein4D


def _multi_indices(dimensions: int, limit: int, total: int):
    return sorted(
        (
            alpha
            for alpha in itertools.product(range(limit), repeat=dimensions)
            if sum(alpha) == total
        ),
        reverse=True,
    )


def _interpolate(dimensions: int, degree: int, groups):
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
) -> Float[Bernstein, "*batch"]:
    """Return :class:`Bernstein` of degree 1 from ``f[..., v_x]``.

    ``f`` has shape ``(*batch, 2)``.  ``f[..., 0]`` is $f(0)$ and
    ``f[..., 1]`` is $f(1)$.  The returned controls have the same shape, with
    ``c[..., i_x]`` corresponding to the same $x$ order as ``v_x``.
    """
    return _interpolate(1, 1, (f,))


@jaxtyped(typechecker=beartype)
def linear_interpolate_2d(
    f: Float[jax.Array, "*batch 2 2"],
) -> Float[Bernstein2D, "*batch"]:
    """Return :class:`Bernstein2D` of degree $(1,1)$ from ``f[..., v_x, v_y]``.

    ``f`` has shape ``(*batch, 2, 2)``.  Its final axes are vertices in
    ``x, y`` order: ``f[..., vx, vy] = f(vx, vy)``.  Output
    ``c[..., ix, iy]`` uses the identical ``x, y`` axis order.
    """
    return _interpolate(2, 1, (f,))


@jaxtyped(typechecker=beartype)
def linear_interpolate_3d(
    f: Float[jax.Array, "*batch 2 2 2"],
) -> Float[Bernstein3D, "*batch"]:
    """Return :class:`Bernstein3D` of degree $(1,1,1)$.

    ``f[..., vx, vy, vz]`` has shape ``(*batch, 2, 2, 2)`` and stores
    $f(v_x,v_y,v_z)$.  Returned ``c[..., ix, iy, iz]`` preserves ``x,y,z``.
    """
    return _interpolate(3, 1, (f,))


@jaxtyped(typechecker=beartype)
def linear_interpolate_4d(
    f: Float[jax.Array, "*batch 2 2 2 2"],
) -> Float[Bernstein4D, "*batch"]:
    """Return :class:`Bernstein4D` of degree $(1,1,1,1)$.

    ``f[..., vx, vy, vz, vw]`` has shape ``(*batch, 2, 2, 2, 2)``.  Returned
    ``c[..., ix, iy, iz, iw]`` preserves the exact ``x,y,z,w`` order.
    """
    return _interpolate(4, 1, (f,))


@jaxtyped(typechecker=beartype)
def hermite_interpolate_1d(
    f: Float[jax.Array, "*batch 2"], d1: Float[jax.Array, "*batch 2"]
) -> Float[Bernstein, "*batch"]:
    """Return degree-3 1D controls from values and first endpoint derivatives.

    ``f[..., vx]`` and ``d1[..., vx]`` both have shape ``(*batch, 2)``;
    they store $f(v_x)$ and $f_x(v_x)$.  Output ``c[..., ix]`` is in $x$
    order.
    """
    return _interpolate(1, 3, (f, d1))


@jaxtyped(typechecker=beartype)
def hermite_interpolate_2d(
    f: Float[jax.Array, "*batch 2 2"],
    d1: Float[jax.Array, "*batch 2 2 2"],
    d2: Float[jax.Array, "*batch 2 2"],
) -> Float[Bernstein2D, "*batch"]:
    """Interpolate $f(x,y)$ with a degree-$(3,3)$ :class:`Bernstein2D`.

    **Inputs:** vertex axes are ``vx, vy``: ``f[...,vx,vy]=f(vx,vy)``.
    ``d1[...,vx,vy,0/1]=(f_x,f_y)`` and
    ``d2[...,vx,vy]=f_xy``.
    **Returns:** ``bpoly.order == [3, 3]``; ``bpoly.c[...,ix,iy]`` is in
    the same ``x,y`` order as the vertex axes.
    """
    return _interpolate(2, 3, (f, d1, d2))


def hermite_interpolate_3d(f, d1, d2, d3):
    """Return degree-(3,3,3) controls in ``x,y,z`` order.

    ``f`` and ``d3=f_xyz`` have shape ``(*batch,2,2,2)``.
    ``d1[...,0/1/2]=f_x/f_y/f_z`` and
    ``d2[...,0/1/2]=f_xy/f_xz/f_yz`` have final derivative-kind axes.
    """
    return _interpolate(3, 3, (f, d1, d2, d3))


def hermite_interpolate_4d(f, d1, d2, d3, d4):
    """Interpolate $f(x,y,z,w)$ with a degree-$(3,3,3,3)$ :class:`Bernstein4D`.

    **Inputs:** ``f[...,vx,vy,vz,vw]=f(vx,vy,vz,vw)``.
    ``d1[...,0..3]=(f_x,f_y,f_z,f_w)``,
    ``d2[...,0..5]=(f_xy,f_xz,f_xw,f_yz,f_yw,f_zw)``,
    ``d3[...,0..3]=(f_xyz,f_xyw,f_xzw,f_yzw)``, and
    ``d4[...,vx,vy,vz,vw]=f_xyzw``.
    **Returns:** ``bpoly.order == [3, 3, 3, 3]``; output
    ``bpoly.c[...,ix,iy,iz,iw]`` is in ``x,y,z,w`` order.
    """
    return _interpolate(4, 3, (f, d1, d2, d3, d4))


def quintic_hermite_interpolate_1d(f, d1, d2):
    """Interpolate $f(x)$ with a degree-5 :class:`Bernstein`.

    **Inputs:** ``f[...,vx]=f(vx)``, ``d1[...,vx]=f_x(vx)``, and
    ``d2[...,vx]=f_xx(vx)``; all have shape ``(*batch,2)``.
    **Returns:** ``bpoly.order == 5``; output ``bpoly.c[...,ix]`` is in
    $x$ order.
    """
    return _interpolate(1, 5, (f, d1, d2))


def quintic_hermite_interpolate_2d(f, d1, d2, d3, d4):
    """Interpolate $f(x,y)$ with a degree-$(5,5)$ :class:`Bernstein2D`.

    **Inputs:** vertex axes are ``vx, vy``: ``f[...,vx,vy]=f(vx,vy)``.
    ``d1[...,vx,vy,0/1]=(f_x,f_y)``,
    ``d2[...,vx,vy,0/1/2]=(f_xx,f_xy,f_yy)``,
    ``d3[...,vx,vy,0/1]=(f_xxy,f_xyy)``, and
    ``d4[...,vx,vy]=f_xxyy``.
    **Returns:** ``bpoly.order == [5, 5]``; ``bpoly.c[...,ix,iy]`` is in
    the same ``x,y`` order as the vertex axes.
    """
    return _interpolate(2, 5, (f, d1, d2, d3, d4))


def quintic_hermite_interpolate_3d(f, d1, d2, d3, d4, d5, d6):
    """Interpolate $f(x,y,z)$ with a degree-$(5,5,5)$ :class:`Bernstein3D`.

    **Inputs:** vertex axes are ``vx, vy, vz``:
    ``f[...,vx,vy,vz]=f(vx,vy,vz)``.
    ``d1[...,vx,vy,vz,0/1/2]=(f_x,f_y,f_z)``,
    ``d2[...,vx,vy,vz,0/1/2/3/4/...]=(f_xx,f_xy,f_xz,f_yy,f_yz,...)``,
    ``d3[...,vx,vy,vz,0/1/2/3/4/...]=(f_xxy,f_xxz,f_xyy,f_xyz,f_xzz,...)``,
    ``d4[...,vx,vy,vz,0/1/2/3/4/...]=(f_xxyy,f_xxyz,f_xxzz,f_xyyz,f_xyzz,...)``,
    ``d5[...,vx,vy,vz,0/1/2]=(f_xxyyz,f_xxyzz,f_xyyzz)``, and
    ``d6[...,vx,vy,vz]=f_xxyyzz``.
    **Returns:** ``bpoly.order == [5, 5, 5]``;
    ``bpoly.c[...,ix,iy,iz]`` is in the same ``x,y,z`` order as the vertex
    axes.
    """
    return _interpolate(3, 5, (f, d1, d2, d3, d4, d5, d6))


def quintic_hermite_interpolate_4d(f, d1, d2, d3, d4, d5, d6, d7, d8):
    """Interpolate $f(x,y,z,w)$ with a degree-$(5,5,5,5)$ :class:`Bernstein4D`.

    **Inputs:** vertex axes are ``vx, vy, vz, vw``:
    ``f[...,vx,vy,vz,vw]=f(vx,vy,vz,vw)``.
    ``d1[...,vx,vy,vz,vw,0/1/2/3]=(f_x,f_y,f_z,f_w)``,
    ``d2[...,vx,vy,vz,vw,0/1/2/3/4/...]=(f_xx,f_xy,f_xz,f_xw,f_yy,...)``,
    ``d3[...,vx,vy,vz,vw,0/1/2/3/4/...]=(f_xxy,f_xxz,f_xxw,f_xyy,f_xyz,...)``,
    ``d4[...,vx,vy,vz,vw,0/1/2/3/4/...]=(f_xxyy,f_xxyz,f_xxyw,f_xxzz,f_xxzw,...)``,
    ``d5[...,vx,vy,vz,vw,0/1/2/3/4/...]=(f_xxyyz,f_xxyyw,f_xxyzz,f_xxyzw,f_xxyww,...)``,
    ``d6[...,vx,vy,vz,vw,0/1/2/3/4/...]=(f_xxyyzz,f_xxyyzw,f_xxyyww,f_xxyzzw,f_xxyzww,...)``,
    ``d7[...,vx,vy,vz,vw,0/1/2/3]=(f_xxyyzzw,f_xxyyzww,f_xxyzzww,f_xyyzzww)``,
    and ``d8[...,vx,vy,vz,vw]=f_xxyyzzww``.
    **Returns:** ``bpoly.order == [5, 5, 5, 5]``;
    ``bpoly.c[...,ix,iy,iz,iw]`` is in the same ``x,y,z,w`` order as the
    vertex axes.
    """
    return _interpolate(4, 5, (f, d1, d2, d3, d4, d5, d6, d7, d8))
