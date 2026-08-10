r"""Shared operations for tensor-product Bernstein polynomials.

For $d$ parameters, write $\mathbf{u}=(u_0,\ldots,u_{d-1})\in[0,1]^d$,
the degree vector as $\mathbf{n}=(n_0,\ldots,n_{d-1})$, and a multi-index as
$\mathbf{i}=(i_0,\ldots,i_{d-1})$. This module represents

$$
p(\mathbf{u}) =
\sum_{0\leq i_a\leq n_a}
c_{\mathbf{i}}\prod_{a=0}^{d-1}B_{i_a}^{n_a}(u_a),
$$

where $B_i^n$ is the one-dimensional Bernstein basis defined in
``xbernstein.bpoly``. The final $d$ coefficient-array axes correspond, in
order, to the components of $\mathbf{i}$; all earlier axes are batch or value
axes.

Batch dimensions
----------------

For coefficient shape ``(*batch, n_0 + 1, ..., n_{d-1} + 1)``, every leading
index identifies an independent tensor-product polynomial. Coordinate arrays
``u_0, ..., u_{d-1}`` broadcast to a shared shape ``*points``; evaluation then
returns shape ``(*batch, *points)``. The final $d$ coefficient axes are never
batch axes.

```python
>>> import jax.numpy as jnp
>>> from xbernstein import Bernstein2D
>>> patches = Bernstein2D(jnp.zeros((2, 2, 3)))
>>> patches(jnp.zeros((3, 1)), jnp.zeros((1, 4))).shape
(2, 3, 4)
>>> batched_patches = Bernstein2D(jnp.zeros((2, 3, 2, 3)))
>>> batched_patches(jnp.zeros((4, 1)), jnp.zeros((1, 5))).shape
(2, 3, 4, 5)
>>> try:
...     patches(jnp.zeros((2,)), jnp.zeros((3,)))
... except ValueError as error:
...     type(error).__name__
'ValueError'
```
"""

from typing import ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jss
from jaxtyping import Float, Int

from .bernstein import Bernstein


class _TensorBernstein(eqx.Module):
    r"""Implement $p(\mathbf{u})$ in the tensor-product Bernstein basis.

    Each subclass fixes the parameter dimension $d$. If ``c`` has shape
    ``(*batch, n_0 + 1, ..., n_{d-1} + 1)``, its trailing axes store
    $c_{\mathbf{i}}$ and its leading axes identify independently evaluated
    polynomials. Every axis argument refers to the parameter index $a$ in
    $\mathbf{u}$, never to a leading batch axis.

    Batch dimensions
    ----------------

    The shared implementation broadcasts only leading coefficient axes for
    arithmetic. Trailing $d$ axes are degree axes, so they are aligned by
    polynomial degree rather than raw-array broadcasting.

    For example, these raw arrays cannot be added:

    ```python
    >>> import jax.numpy as jnp
    >>> try:
    ...     jnp.zeros((2, 2, 3)) + jnp.zeros((3, 2))
    ... except ValueError as error:
    ...     type(error).__name__
    'ValueError'
    ```

    The corresponding 2D Bernstein objects do add. The first has batch shape
    ``(2,)`` and degree vector $(1,2)$; the second has no batch axes and
    degree vector $(2,1)$. The scalar-batch polynomial broadcasts and each
    degree axis is elevated independently:

    ```python
    >>> first = Bernstein2D(jnp.zeros((2, 2, 3)))
    >>> second = Bernstein2D(jnp.zeros((3, 2)))
    >>> (first + second).c.shape
    (2, 3, 3)
    ```

    The same rule applies when both operands have multiple leading batch
    axes. Their batch shapes are right-aligned independently of their degree
    axes:

    ```python
    >>> left = Bernstein2D(jnp.zeros((2, 1, 2, 3)))   # Batch shape (2, 1).
    >>> right = Bernstein2D(jnp.zeros((1, 3, 3, 2)))  # Batch shape (1, 3).
    >>> (left + right).c.shape
    (2, 3, 3, 3)
    ```

    This convention applies to ``+``, ``-``, and ``*``. Coordinate arrays,
    conversely, create evaluation axes rather than pairing with batch axes:

    ```python
    >>> first(jnp.zeros((2,)), 0.5).shape
    (2, 2)
    >>> left(jnp.zeros((4, 1)), jnp.zeros((1, 5))).shape
    (2, 1, 4, 5)
    ```

    True leading-batch incompatibility still raises an error:

    ```python
    >>> try:
    ...     first + Bernstein2D(jnp.zeros((3, 2, 3)))
    ... except ValueError as error:
    ...     type(error).__name__
    'ValueError'
    ```
    """

    c: Float[jax.Array, "..."]
    parameter_dimensions: ClassVar[int]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the coefficient-array axes preceding the multi-index $\mathbf{i}$."""
        return self.c.shape[: -self.parameter_dimensions]

    @property
    def order(self) -> Int[jax.Array, " dim"]:
        r"""Return the parameter-wise degree vector $\mathbf{n}$.

        If $d=$ ``parameter_dimensions``, the final coefficient axes store
        the multi-index components $i_0,\ldots,i_{d-1}$. Their lengths obey

        $$
        \bigl(\text{``c.shape[-d]``},\ldots,\text{``c.shape[-1]``}\bigr)
        = \mathbf{n} + \mathbf{1}
        = \text{``order``} + \mathbf{1}.
        $$

        The returned JAX integer array has shape ``(d,)`` in the same axis
        order as the polynomial parameters.
        """
        return jnp.asarray(self.c.shape[-self.parameter_dimensions :]) - 1

    @property
    def dtype(self) -> str:
        """Return the scalar field used for every tensor coefficient $c_{\mathbf{i}}$."""
        return str(self.c.dtype)

    def _axis_index(self, axis: int) -> int:
        r"""Map parameter index $a$ to the matching trailing coefficient-array axis.

        The parameter factor $B_{i_a}^{n_a}(u_a)$ is stored on array axis
        ``c.ndim - d + a``. This helper validates $0\leq a<d$ before returning
        that axis.
        """
        if not 0 <= axis < self.parameter_dimensions:
            raise ValueError(
                f"axis must be in [0, {self.parameter_dimensions}), got {axis}"
            )
        return self.c.ndim - self.parameter_dimensions + axis

    def _lower_dimension(self, coefficients: jax.Array):
        r"""Construct the Bernstein representation after removing one $u_a$ factor.

        The input coefficients retain their original order except that one
        multi-index component has been eliminated. Thus a $d$-dimensional
        tensor polynomial becomes the package's $(d-1)$-dimensional class.
        """
        if self.parameter_dimensions == 2:
            return Bernstein(coefficients)
        if self.parameter_dimensions == 3:
            from .bernstein_2d import Bernstein2D

            return Bernstein2D(coefficients)
        if self.parameter_dimensions == 4:
            from .bernstein_3d import Bernstein3D

            return Bernstein3D(coefficients)
        raise RuntimeError("tensor Bernstein polynomials require at least 2 dimensions")

    def _elevate_axis(self, c: jax.Array, axis: int, target_degree: int) -> jax.Array:
        r"""Elevate degree $n_a$ to ``target_degree`` along one parameter $u_a$.

        The represented function $p(\mathbf{u})$ is unchanged. For the
        selected factor, each elevation step applies

        $$
        \tilde c_{\ldots,i_a,\ldots}
        =\frac{i_a}{r+1}c_{\ldots,i_a-1,\ldots}
        +\left(1-\frac{i_a}{r+1}\right)c_{\ldots,i_a,\ldots},
        $$

        while every other multi-index component remains fixed.
        """
        axis_index = c.ndim - self.parameter_dimensions + axis
        w = jnp.moveaxis(c, axis_index, -1)
        degree = w.shape[-1] - 1

        for new_degree in range(degree + 1, target_degree + 1):
            i = jnp.arange(1, new_degree, dtype=w.dtype)
            alpha = i / new_degree
            middle = alpha * w[..., :-1] + (1.0 - alpha) * w[..., 1:]
            w = jnp.concatenate([w[..., :1], middle, w[..., -1:]], axis=-1)

        return jnp.moveaxis(w, -1, axis_index)

    def _elevate(self, c: jax.Array, target_degrees: tuple[int, ...]) -> jax.Array:
        r"""Elevate $p$ to the degree vector $\mathbf{N}$ without changing $p(\mathbf{u})$.

        Applies :meth:`_elevate_axis` independently to each component
        $n_a\mapsto N_a$ of the requested degree vector.
        """
        for axis, target_degree in enumerate(target_degrees):
            c = self._elevate_axis(c, axis, target_degree)
        return c

    def _broadcast_coefficients(
        self, c: jax.Array, batch_shape: tuple[int, ...], degree_shape: tuple[int, ...]
    ) -> jax.Array:
        """Broadcast $c_{\mathbf{i}}$ over leading axes without changing its basis axes."""
        return jnp.broadcast_to(c, batch_shape + degree_shape)

    def _aligned_coefficients(self, other: Self) -> tuple[jax.Array, jax.Array]:
        r"""Represent $p$ and $q$ in their shared degree vector before addition.

        The $a$-th common degree is $N_a=\max(n_a,m_a)$. Degree elevation
        preserves both functions, so the returned tensors can be combined
        coefficientwise in the common tensor-product basis.
        """
        d = self.parameter_dimensions
        degrees = tuple(
            max(self.c.shape[-d + axis], other.c.shape[-d + axis]) - 1
            for axis in range(d)
        )
        batch_shape = jnp.broadcast_shapes(self.shape, other.shape)
        degree_shape = tuple(degree + 1 for degree in degrees)
        c1 = self._broadcast_coefficients(
            self._elevate(self.c, degrees), batch_shape, degree_shape
        )
        c2 = self._broadcast_coefficients(
            self._elevate(other.c, degrees), batch_shape, degree_shape
        )
        return c1, c2

    def __add__(self, other: Self) -> Self:
        r"""Return $p+q$ in the componentwise maximum-degree basis.

        Each parameter degree is elevated to
        $N_a=\max(n_a,m_a)$ before corresponding coefficients are added.

        See :class:`_TensorBernstein` for the common arithmetic batch
        convention and executable examples.
        """
        if not isinstance(other, type(self)):
            raise TypeError(
                f"{type(self).__name__} arithmetic requires another "
                f"{type(self).__name__}"
            )
        c1, c2 = self._aligned_coefficients(other)
        return type(self)(c1 + c2)

    def __sub__(self, other: Self) -> Self:
        r"""Return $p-q$ in the componentwise maximum-degree basis.

        Each parameter degree is elevated to
        $N_a=\max(n_a,m_a)$ before corresponding coefficients are subtracted.

        See :class:`_TensorBernstein` for the common arithmetic batch
        convention and executable examples.
        """
        if not isinstance(other, type(self)):
            raise TypeError(
                f"{type(self).__name__} arithmetic requires another "
                f"{type(self).__name__}"
            )
        c1, c2 = self._aligned_coefficients(other)
        return type(self)(c1 - c2)

    def __mul__(self, other: Self) -> Self:
        r"""Return the pointwise product $r(\mathbf{u})=p(\mathbf{u})q(\mathbf{u})$.

        If $p$ and $q$ have degree vectors $\mathbf{n}$ and $\mathbf{m}$,
        then $r$ has degree vector $\mathbf{n}+\mathbf{m}$. For a multi-index
        $\mathbf{k}$, its coefficient is

        $$
        r_{\mathbf{k}} =
        \sum_{\mathbf{i}+\mathbf{j}=\mathbf{k}}
        c_{\mathbf{i}}d_{\mathbf{j}}
        \prod_{a=0}^{d-1}
        \frac{\binom{n_a}{i_a}\binom{m_a}{j_a}}
        {\binom{n_a+m_a}{k_a}}.
        $$

        The implementation forms this separable product factor and accumulates
        all pairs of multi-indices with the same $\mathbf{k}$.

        See :class:`_TensorBernstein` for the common arithmetic batch
        convention and executable examples.
        """
        if not isinstance(other, type(self)):
            raise TypeError(
                f"{type(self).__name__} arithmetic requires another "
                f"{type(self).__name__}"
            )

        d = self.parameter_dimensions
        batch_shape = jnp.broadcast_shapes(self.shape, other.shape)
        n_shape = self.c.shape[-d:]
        m_shape = other.c.shape[-d:]
        c1 = self._broadcast_coefficients(self.c, batch_shape, n_shape)
        c2 = self._broadcast_coefficients(other.c, batch_shape, m_shape)

        grid_shape = n_shape + m_shape
        scale = jnp.ones(grid_shape, dtype=c1.dtype)
        flat_indices = jnp.zeros(grid_shape, dtype=jnp.int32)
        output_shape = tuple(n + m - 1 for n, m in zip(n_shape, m_shape))
        strides = tuple(
            int(jnp.prod(jnp.asarray(output_shape[axis + 1 :]))) for axis in range(d)
        )

        for axis, (n_size, m_size, stride) in enumerate(zip(n_shape, m_shape, strides)):
            i = jnp.arange(n_size).reshape(
                (1,) * axis + (n_size,) + (1,) * (2 * d - axis - 1)
            )
            j = jnp.arange(m_size).reshape(
                (1,) * (d + axis) + (m_size,) + (1,) * (d - axis - 1)
            )
            n = n_size - 1
            m = m_size - 1
            scale = scale * jss.comb(n, i) * jss.comb(m, j) / jss.comb(n + m, i + j)
            flat_indices = flat_indices + (i + j) * stride

        products = (
            c1.reshape(batch_shape + n_shape + (1,) * d)
            * c2.reshape(batch_shape + (1,) * d + m_shape)
            * scale
        )
        result = jnp.zeros(batch_shape + output_shape, dtype=products.dtype).reshape(
            batch_shape + (-1,)
        )
        result = result.at[..., flat_indices.reshape(-1)].add(
            products.reshape(batch_shape + (-1,))
        )
        return type(self)(result.reshape(batch_shape + output_shape))

    def deriv(self, m: int = 1, axis: int = 0) -> Self:
        r"""Return the partial derivative $\partial_{u_a}^m p$.

        With $a=$ ``axis`` and selected degree $n_a$, one derivative changes
        only that multi-index component:

        $$
        d_{i_0,\ldots,i_a,\ldots,i_{d-1}}
        =n_a\left(
        c_{i_0,\ldots,i_a+1,\ldots,i_{d-1}}
        -c_{i_0,\ldots,i_a,\ldots,i_{d-1}}\right).
        $$

        The transformation is repeated ``m`` times. If $m>n_a$, the result is
        the zero polynomial with degree $0$ in $u_a$.

        Batch dimensions
        ----------------

        Partial differentiation preserves leading axes and every unselected
        parameter-degree axis:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import Bernstein2D
        >>> Bernstein2D(jnp.zeros((2, 3, 4))).deriv(axis=1).c.shape
        (2, 3, 3)
        ```
        """
        axis_index = self._axis_index(axis)
        w = jnp.moveaxis(self.c, axis_index, -1)
        degree = w.shape[-1] - 1

        if m > degree:
            w = jnp.zeros(w.shape[:-1] + (1,), dtype=w.dtype)
        else:
            for current_degree in range(degree, degree - m, -1):
                w = current_degree * (w[..., 1:] - w[..., :-1])

        return type(self)(jnp.moveaxis(w, -1, axis_index))

    def int(self, k: float = 0.0, axis: int = 0) -> Self:
        r"""Return $P$ satisfying $\partial_{u_a}P=p$ with $P|_{u_a=0}=k$.

        For every fixed set of the remaining coordinates, this is the
        one-dimensional Bernstein antiderivative along $u_a$. If the selected
        degree is $n_a$, its coefficients satisfy

        $$
        C_{\ldots,0,\ldots}=k,\qquad
        C_{\ldots,i_a+1,\ldots}
        =k+\frac{1}{n_a+1}
        \sum_{j=0}^{i_a}c_{\ldots,j,\ldots}.
        $$

        The constant ``k`` broadcasts over all unselected coefficient axes.

        Batch dimensions
        ----------------

        Integration preserves leading axes and adds one control point only on
        the selected parameter axis:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import Bernstein2D
        >>> Bernstein2D(jnp.zeros((2, 3, 4))).int(axis=0).c.shape
        (2, 4, 4)
        ```
        """
        axis_index = self._axis_index(axis)
        w = jnp.moveaxis(self.c, axis_index, -1)
        degree = w.shape[-1] - 1
        constant = jnp.broadcast_to(jnp.asarray(k, dtype=w.dtype), w.shape[:-1])
        integrated = jnp.concatenate(
            [
                constant[..., None],
                constant[..., None] + jnp.cumsum(w, axis=-1) / (degree + 1),
            ],
            axis=-1,
        )
        return type(self)(jnp.moveaxis(integrated, -1, axis_index))

    def __call__(self, *ts: Float[jax.Array, "..."]) -> jax.Array:
        r"""Evaluate $p(\mathbf{u})$ at ``(u_0, ..., u_{d-1})``.

        For each coordinate $u_a$, De Casteljau repeatedly replaces adjacent
        entries along the $i_a$ axis by

        $$
        c_{\ldots,i_a,\ldots}^{(r+1)}
        =(1-u_a)c_{\ldots,i_a,\ldots}^{(r)}
        +u_a c_{\ldots,i_a+1,\ldots}^{(r)}.
        $$

        Reducing every parameter axis evaluates the tensor-product expansion
        on $[0,1]^d$. Parameter arrays broadcast together to form evaluation
        axes after the leading coefficient axes.

        Batch dimensions
        ----------------

        Broadcast-compatible coordinate arrays form shared evaluation axes
        after leading coefficient axes:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import Bernstein2D
        >>> patch = Bernstein2D(jnp.zeros((2, 2, 3)))
        >>> patch(jnp.linspace(0.0, 1.0, 4), 0.5).shape
        (2, 4)
        >>> batched = Bernstein2D(jnp.zeros((2, 3, 2, 3)))
        >>> batched(jnp.zeros((4, 1)), jnp.zeros((1, 5))).shape
        (2, 3, 4, 5)
        ```
        """
        if len(ts) != self.parameter_dimensions:
            raise TypeError(
                f"expected {self.parameter_dimensions} parameter arrays, got {len(ts)}"
            )

        d = self.parameter_dimensions
        batch_dimensions = self.c.ndim - d
        parameters = tuple(jnp.asarray(t, dtype=self.c.dtype) for t in ts)
        point_shape = jnp.broadcast_shapes(*(t.shape for t in parameters))
        parameters = tuple(jnp.broadcast_to(t, point_shape) for t in parameters)
        w = self.c.reshape(self.shape + (1,) * len(point_shape) + self.c.shape[-d:])

        for axis in range(d - 1, -1, -1):
            t = parameters[axis].reshape(
                (1,) * batch_dimensions + point_shape + (1,) * (axis + 1)
            )
            for _ in range(w.shape[-1] - 1):
                w = (1.0 - t) * w[..., :-1] + t * w[..., 1:]
            w = w[..., 0]

        return w

    def split(
        self, t: Float[jax.Array, "..."] = jnp.array(0.5), axis: int = 0
    ) -> tuple[Self, Self]:
        r"""Split along $u_a=t$ and reparameterize both pieces to $[0,1]^d$.

        The first result represents

        $$
        p(u_0,\ldots,ts,\ldots,u_{d-1}),
        $$

        and the second represents

        $$
        p(u_0,\ldots,t+(1-t)s,\ldots,u_{d-1}),
        \qquad s\in[0,1].
        $$

        Their selected-axis control polygons are the first and reversed-last
        entries of the corresponding De Casteljau table.

        Batch dimensions
        ----------------

        The split value may have the leading batch shape. Both results keep
        the same coefficient shape as the input:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import Bernstein2D
        >>> lower, upper = Bernstein2D(jnp.zeros((2, 2, 3))).split(jnp.array([0.25, 0.75]))
        >>> lower.c.shape, upper.c.shape
        ((2, 2, 3), (2, 2, 3))
        ```
        """
        axis_index = self._axis_index(axis)
        w = jnp.moveaxis(self.c, axis_index, -1)
        t = jnp.asarray(t, dtype=w.dtype)
        t = jnp.broadcast_to(t, self.shape).reshape(
            self.shape + (1,) * self.parameter_dimensions
        )

        left = [w[..., 0]]
        right = [w[..., -1]]
        for _ in range(w.shape[-1] - 1):
            w = (1.0 - t) * w[..., :-1] + t * w[..., 1:]
            left.append(w[..., 0])
            right.append(w[..., -1])

        left_coefficients = jnp.moveaxis(jnp.stack(left, axis=-1), -1, axis_index)
        right_coefficients = jnp.moveaxis(
            jnp.stack(right[::-1], axis=-1), -1, axis_index
        )
        return type(self)(left_coefficients), type(self)(right_coefficients)

    def slice(self, value: Float[jax.Array, "..."], axis: int = 0):
        r"""Restrict $p$ to the hyperplane $u_a=v$ and remove coordinate $u_a$.

        For ``axis`` $=a$ and ``value`` $=v$, De Casteljau evaluates the
        selected factor $B_{i_a}^{n_a}(v)$. The returned $(d-1)$-dimensional
        polynomial is

        $$
        q(u_0,\ldots,u_{a-1},u_{a+1},\ldots,u_{d-1})
        =p(u_0,\ldots,u_{a-1},v,u_{a+1},\ldots,u_{d-1}).
        $$

        Its coefficient axes retain the order of the unselected parameters.

        Batch dimensions
        ----------------

        ``value`` broadcasts against leading coefficient axes. The result
        keeps those leading axes and removes exactly one trailing
        parameter-degree axis:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import Bernstein2D
        >>> Bernstein2D(jnp.zeros((2, 2, 3))).slice(jnp.array([0.25, 0.75]), axis=0).c.shape
        (2, 3)
        >>> Bernstein2D(jnp.zeros((2, 3, 2, 3))).slice(jnp.zeros((2, 1)), axis=0).c.shape
        (2, 3, 3)
        ```
        """
        self._axis_index(axis)
        value = jnp.asarray(value, dtype=self.c.dtype)
        dimensions = self.parameter_dimensions
        degrees = self.c.shape[-dimensions:]
        batch_shape = jnp.broadcast_shapes(self.shape, value.shape)
        coefficients = self._broadcast_coefficients(self.c, batch_shape, degrees)
        axis_index = len(batch_shape) + axis
        w = jnp.moveaxis(coefficients, axis_index, -1)
        weight = jnp.broadcast_to(value, batch_shape).reshape(
            batch_shape + (1,) * dimensions
        )

        for _ in range(w.shape[-1] - 1):
            w = (1.0 - weight) * w[..., :-1] + weight * w[..., 1:]

        return self._lower_dimension(w[..., 0])

    def integrate_out(self, axis: int = 0):
        r"""Integrate $p$ over $u_a\in[0,1]$ and remove coordinate $u_a$.

        Because every degree-$n_a$ Bernstein basis function has equal integral,

        $$
        \int_0^1 B_{i_a}^{n_a}(u_a)\,du_a=\frac{1}{n_a+1},
        $$

        the returned coefficient tensor is

        $$
        q_{\mathbf{i}_{\neg a}}
        =\frac{1}{n_a+1}
        \sum_{i_a=0}^{n_a}c_{\mathbf{i}}.
        $$

        Here $\mathbf{i}_{\neg a}$ is the multi-index with component $a$
        omitted, and the remaining coefficient axes keep that order.

        Batch dimensions
        ----------------

        The integral is evaluated independently at every leading index. It
        preserves leading axes and removes the selected trailing
        parameter-degree axis:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import Bernstein3D
        >>> Bernstein3D(jnp.zeros((2, 2, 3, 4))).integrate_out(axis=1).c.shape
        (2, 2, 4)
        ```
        """
        axis_index = self._axis_index(axis)
        order = self.c.shape[axis_index]
        coefficients = jnp.sum(self.c, axis=axis_index) / order
        return self._lower_dimension(coefficients)

    def segment(
        self, start: Float[jax.Array, "..."], end: Float[jax.Array, "..."]
    ) -> Bernstein:
        r"""Return the 1D Bernstein restriction of $p$ to an affine segment.

        With endpoints $\mathbf{a}=$ ``start`` and $\mathbf{b}=$ ``end``, the
        returned polynomial represents

        $$
        q(t)=p\bigl(\mathbf{a}+t(\mathbf{b}-\mathbf{a})\bigr),
        \qquad t\in[0,1].
        $$

        For each tensor axis $a$, the control points of
        $B_{i_a}^{n_a}(a_a+t(b_a-a_a))$ are obtained from its blossom. Its
        $k$-th point evaluates the De Casteljau table using $n_a-k$ copies of
        $a_a$ and $k$ copies of $b_a$. The resulting univariate factors are
        multiplied with the Bernstein product identity, yielding degree
        $\sum_a n_a$ without solving a linear system.

        Batch dimensions
        ----------------

        ``start`` and ``end`` may each have shape ``(*batch, d)`` and
        broadcast with the leading coefficient axes. The result keeps the
        broadcast leading shape and replaces all parameter-degree axes with
        one univariate coefficient axis:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import Bernstein2D
        >>> patch = Bernstein2D(jnp.zeros((2, 2, 3)))
        >>> patch.segment(jnp.zeros((2, 2)), jnp.ones((2, 2))).c.shape
        (2, 4)
        >>> patch = Bernstein2D(jnp.zeros((2, 3, 2, 3)))
        >>> patch.segment(jnp.zeros((2, 1, 2)), jnp.ones((1, 3, 2))).c.shape
        (2, 3, 4)
        ```
        """
        start = jnp.asarray(start, dtype=self.c.dtype)
        end = jnp.asarray(end, dtype=self.c.dtype)
        dimensions = self.parameter_dimensions
        if (
            start.ndim == 0
            or end.ndim == 0
            or start.shape[-1] != dimensions
            or end.shape[-1] != dimensions
        ):
            raise ValueError(
                "start and end must have a final dimension of "
                f"{dimensions}, got {start.shape} and {end.shape}"
            )

        batch_shape = jnp.broadcast_shapes(self.shape, start.shape[:-1], end.shape[:-1])
        degrees = self.c.shape[-dimensions:]
        coefficients = self._broadcast_coefficients(self.c, batch_shape, degrees)
        starts = jnp.broadcast_to(start, batch_shape + (dimensions,))
        ends = jnp.broadcast_to(end, batch_shape + (dimensions,))

        # Reparameterize each Bernstein basis along the segment with its blossom.
        # The k-th control point uses `start` n-k times and `end` k times.
        for axis, axis_size in enumerate(degrees):
            axis_degree = axis_size - 1
            basis = jnp.broadcast_to(
                jnp.eye(axis_size, dtype=self.c.dtype),
                batch_shape + (axis_size, axis_size),
            )
            reparameterized = []
            for end_count in range(axis_size):
                w = basis
                for parameter in [starts[..., axis]] * (axis_degree - end_count) + [
                    ends[..., axis]
                ] * end_count:
                    weight = parameter.reshape(batch_shape + (1, 1))
                    w = (1.0 - weight) * w[..., :-1, :] + weight * w[..., 1:, :]
                reparameterized.append(w[..., 0, :])
            transform = jnp.stack(reparameterized, axis=-2)

            axis_index = len(batch_shape) + axis
            coefficients = jnp.moveaxis(coefficients, axis_index, -1)
            other_dimensions = coefficients.ndim - len(batch_shape) - 1
            transform = transform.reshape(
                batch_shape + (1,) * other_dimensions + transform.shape[-2:]
            )
            coefficients = jnp.sum(coefficients[..., None, :] * transform, axis=-1)
            coefficients = jnp.moveaxis(coefficients, -1, axis_index)

        degree = sum(axis_size - 1 for axis_size in degrees)
        if degree == 0:
            return Bernstein(coefficients.reshape(batch_shape + (1,)))

        multi_indices = jnp.indices(degrees)
        total_indices = jnp.sum(multi_indices, axis=0)
        scale = jnp.ones(degrees, dtype=self.c.dtype)
        for axis, axis_size in enumerate(degrees):
            scale = scale * jss.comb(axis_size - 1, multi_indices[axis])
        scale = scale / jss.comb(degree, total_indices)

        control_points = (
            jnp.zeros(batch_shape + (degree + 1,), dtype=self.c.dtype)
            .at[..., total_indices]
            .add(coefficients * scale)
        )
        return Bernstein(control_points)
