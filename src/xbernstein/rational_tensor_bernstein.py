r"""Positive-weight scalar rational tensor-product Bernstein functions."""

import itertools
import math
from typing import ClassVar

import equinox as eqx
from beartype import beartype
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, jaxtyped

from ._tensor_bernstein import (
    _evaluate_tensor_coefficients,
    _split_tensor_coefficients,
)
from .bernstein_2d import Bernstein2D
from .bernstein_3d import Bernstein3D
from .bernstein_4d import Bernstein4D
from .rational_bernstein import RationalBernstein, _from_homogeneous


def _homogeneous_component(
    h: Float[jax.Array, "..."], dimensions: int, index: int
) -> Float[jax.Array, "..."]:
    return jnp.take(h, index, axis=-(dimensions + 1))


@jaxtyped(typechecker=beartype)
class _RationalTensorBernstein(eqx.Module):
    r"""Represent a positive-weight rational tensor-product Bernstein function.

    概要
    ----
    The domain is $[0,1]^d$ and the represented function is the rational
    tensor-product polynomial $R(\mathbf{u})=N(\mathbf{u})/D(\mathbf{u})$.

    数学的表現
    ----------

    For parameter vector
    $\mathbf{u}=(u_0,\ldots,u_{d-1})$, degree vector
    $\mathbf{n}=(n_0,\ldots,n_{d-1})$, and multi-index
    $\mathbf{i}=(i_0,\ldots,i_{d-1})$, define

    $$
    B_{\mathbf{i}}^{\mathbf{n}}(\mathbf{u})
    =\prod_{a=0}^{d-1}B_{i_a}^{n_a}(u_a).
    $$

    The represented scalar function is

    $$
    R(\mathbf{u})=\frac{N(\mathbf{u})}{D(\mathbf{u})}
    =\frac{\displaystyle\sum_{\mathbf{i}}
    w_{\mathbf{i}}c_{\mathbf{i}}
    B_{\mathbf{i}}^{\mathbf{n}}(\mathbf{u})}
    {\displaystyle\sum_{\mathbf{i}}
    w_{\mathbf{i}}B_{\mathbf{i}}^{\mathbf{n}}(\mathbf{u})},
    \qquad w_{\mathbf{i}}>0.
    $$

    Because the tensor-product basis is nonnegative and forms a partition of
    unity on $[0,1]^d$, $D(\mathbf{u})>0$ throughout the domain.

    配列表現
    ----------

    For arrays ``values`` and ``weights`` of shape
    ``(*batch, n_0 + 1, ..., n_{d-1} + 1)``, the stored homogeneous array has
    shape ``(*batch, 2, n_0 + 1, ..., n_{d-1} + 1)`` and satisfies

    $$
    h_{0,\mathbf{i}}=w_{\mathbf{i}}c_{\mathbf{i}},\qquad
    h_{1,\mathbf{i}}=w_{\mathbf{i}}.
    $$

    The length-2 homogeneous axis precedes the final $d$ parameter-degree
    axes. All earlier axes are independent batch axes. The leading shapes of
    ``values`` and ``weights`` broadcast; their final $d$ shapes must agree.

    評価
    ----

    Only axes before the homogeneous and parameter-degree axes are batch
    axes. For example, these inputs broadcast to batch shape ``(2, 4)`` while
    retaining degree vector $(1,2)$:

    ```python
    >>> import jax.numpy as jnp
    >>> from xbernstein import RationalBernstein2D
    >>> surfaces = RationalBernstein2D(
    ...     jnp.zeros((2, 1, 2, 3)),
    ...     jnp.ones((1, 4, 2, 3)),
    ... )
    >>> surfaces.shape, surfaces.h.shape
    ((2, 4), (2, 4, 2, 2, 3))

    ```

    演算
    ----
    Arithmetic broadcasts only these leading axes. Parameter-degree axes are
    combined by Bernstein algebra, not raw-array broadcasting. Rational
    addition, subtraction, and multiplication use denominator products, so
    all three add the two degree vectors componentwise:

    ```python
    >>> left = RationalBernstein2D(jnp.zeros((2, 2, 3)), jnp.ones((2, 3)))
    >>> right = RationalBernstein2D(jnp.zeros((3, 2)), jnp.ones((3, 2)))
    >>> result = left + right
    >>> result.shape
    (2,)
    >>> result.order.tolist()
    [3, 3]

    ```

    The operators ``+``, ``-``, and ``*`` use right-aligned broadcasting for
    every leading axis. Incompatible true batch shapes raise ``ValueError``.
    Coordinate arrays behave differently: they broadcast with one another to
    create evaluation axes after the coefficient batch axes:

    ```python
    >>> surfaces(jnp.zeros((5, 1)), jnp.zeros((1, 6))).shape
    (2, 4, 5, 6)
    >>> first = RationalBernstein2D(jnp.zeros((2, 2, 3)), jnp.ones((2, 2, 3)))
    >>> second = RationalBernstein2D(jnp.zeros((3, 2, 3)), jnp.ones((3, 2, 3)))
    >>> try:
    ...     first + second
    ... except ValueError as error:
    ...     type(error).__name__
    'ValueError'

    ```

    数学的注意点
    ------------
    Partial derivatives are independent-coordinate derivatives, whereas a
    simplex barycentric derivative would preserve a sum constraint.  Strictly
    positive weights imply $D(\mathbf{u})>0$ on the cube.  Batch axes precede the
    homogeneous axis and the final ``d`` coefficient axes; they are not degree
    axes.

    Integration
    -----------

    ``integrate_out()`` is deliberately unavailable. In general,

    $$
    \int_0^1
    \frac{N(u_a,\mathbf{u}_{\neg a})}
         {D(u_a,\mathbf{u}_{\neg a})}\,du_a
    $$

    may contain logarithmic or inverse-trigonometric terms and therefore
    cannot be represented exactly by a lower-dimensional rational Bernstein
    function.
    """

    h: Float[jax.Array, "..."]
    parameter_dimensions: ClassVar[int]
    polynomial_type: ClassVar[type]

    def __init__(
        self,
        values: Float[jax.Array, "..."],
        weights: Float[jax.Array, "..."],
    ):
        r"""Initialize control values $c_{\mathbf{i}}$ and weights $w_{\mathbf{i}}$.

        The final ``parameter_dimensions`` axes of both inputs are the
        tensor-product coefficient axes and must match exactly. Leading axes
        broadcast. Every weight must be strictly positive.

        See :class:`_RationalTensorBernstein` for examples that distinguish
        batch, homogeneous-component, and parameter-degree axes.
        """
        dimensions = self.parameter_dimensions
        values = jnp.asarray(values)
        weights = jnp.asarray(weights)
        if values.ndim < dimensions or weights.ndim < dimensions:
            raise ValueError(
                f"values and weights must have {dimensions} coefficient axes"
            )
        if values.shape[-dimensions:] != weights.shape[-dimensions:]:
            raise ValueError(
                "values and weights must have the same coefficient-axis shape"
            )

        dtype = jnp.result_type(values, weights, jnp.float32)
        values = values.astype(dtype)
        weights = weights.astype(dtype)
        degree_shape = values.shape[-dimensions:]
        batch_shape = jnp.broadcast_shapes(
            values.shape[:-dimensions], weights.shape[:-dimensions]
        )
        values = jnp.broadcast_to(values, batch_shape + degree_shape)
        weights = jnp.broadcast_to(weights, batch_shape + degree_shape)
        weights = eqx.error_if(
            weights,
            jnp.any(weights <= 0.0),
            f"{type(self).__name__} weights must be strictly positive",
        )
        self.h = jnp.stack((weights * values, weights), axis=-(dimensions + 1))

    @classmethod
    def _from_homogeneous(
        cls,
        numerator: Float[jax.Array, "..."],
        denominator: Float[jax.Array, "..."],
    ):
        result = object.__new__(cls)
        axis = -(cls.parameter_dimensions + 1)
        object.__setattr__(result, "h", jnp.stack((numerator, denominator), axis=axis))
        return result

    @property
    def values(self) -> Float[jax.Array, "..."]:
        r"""Return the dehomogenized controls $c_{\mathbf{i}}$.

        They are recovered componentwise from

        $$
        c_{\mathbf{i}}=
        \frac{h_{0,\mathbf{i}}}{h_{1,\mathbf{i}}}.
        $$

        The result has shape
        ``(*batch, n_0 + 1, ..., n_{d-1} + 1)``.
        """
        return self.numerator.c / self.denominator.c

    @property
    def weights(self) -> Float[jax.Array, "..."]:
        r"""Return the positive weights $w_{\mathbf{i}}=h_{1,\mathbf{i}}$.

        Their shape matches :attr:`values`, and positivity guarantees
        $D(\mathbf{u})>0$ on $[0,1]^d$.
        """
        return self.denominator.c

    @property
    def c(self) -> Float[jax.Array, "..."]:
        r"""Return $c_{\mathbf{i}}$ as an alias of :attr:`values`.

        These are dehomogenized controls, not the numerator coefficients
        $w_{\mathbf{i}}c_{\mathbf{i}}$.
        """
        return self.values

    @property
    def w(self) -> Float[jax.Array, "..."]:
        r"""Return $w_{\mathbf{i}}$ as an alias of :attr:`weights`."""
        return self.weights

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Return the leading batch shape ``*batch``.

        This excludes the homogeneous-component axis and all $d$
        parameter-degree axes.
        """
        return self.h.shape[: -(self.parameter_dimensions + 1)]

    @property
    def order(self) -> Int[jax.Array, "dim"]:
        r"""Return the degree vector $\mathbf{n}=(n_0,\ldots,n_{d-1})$.

        If the final coefficient axes have lengths
        ``(n_0 + 1, ..., n_{d-1} + 1)``, this property returns a JAX integer
        array of shape ``(d,)`` in the same parameter order.
        """
        return jnp.asarray(self.h.shape[-self.parameter_dimensions :]) - 1

    @property
    def dtype(self) -> str:
        r"""Return the scalar dtype shared by the homogeneous components."""
        return str(self.h.dtype)

    @property
    def numerator(self):
        r"""Return the tensor Bernstein numerator $N$.

        Its coefficients are

        $$
        n_{\mathbf{i}}=w_{\mathbf{i}}c_{\mathbf{i}}
        =h_{0,\mathbf{i}},
        \qquad
        N(\mathbf{u})=\sum_{\mathbf{i}}
        n_{\mathbf{i}}B_{\mathbf{i}}^{\mathbf{n}}(\mathbf{u}).
        $$

        The return type is the matching polynomial class
        (:class:`Bernstein2D`, :class:`Bernstein3D`, or
        :class:`Bernstein4D`).
        """
        return self.polynomial_type(
            _homogeneous_component(self.h, self.parameter_dimensions, 0)
        )

    @property
    def denominator(self):
        r"""Return the positive tensor Bernstein denominator $D$.

        Its coefficients are

        $$
        d_{\mathbf{i}}=w_{\mathbf{i}}=h_{1,\mathbf{i}},
        \qquad
        D(\mathbf{u})=\sum_{\mathbf{i}}
        d_{\mathbf{i}}B_{\mathbf{i}}^{\mathbf{n}}(\mathbf{u})>0.
        $$
        """
        return self.polynomial_type(
            _homogeneous_component(self.h, self.parameter_dimensions, 1)
        )

    def __call__(
        self, *ts: Float[jax.Array, "..."]
    ) -> Float[jax.Array, "..."]:
        r"""Evaluate $R(\mathbf{u})=N(\mathbf{u})/D(\mathbf{u})$.

        ``ts`` supplies exactly one coordinate array for each component
        $(u_0,\ldots,u_{d-1})$. Numerator and denominator are evaluated by
        tensor-product De Casteljau reduction and then divided.

        Broadcast-compatible coordinate arrays form shared evaluation axes
        after ``self.shape``. For example, coordinate shapes ``(k, 1)`` and
        ``(1, l)`` produce evaluation shape ``(k, l)``.

        Batch dimensions
        ----------------

        A scalar coordinate is shared by every batch item, while array
        coordinates append broadcast evaluation axes:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import RationalBernstein2D
        >>> surfaces = RationalBernstein2D(
        ...     jnp.zeros((2, 3, 2, 3)),
        ...     jnp.ones((2, 3, 2, 3)),
        ... )
        >>> surfaces(0.25, 0.75).shape
        (2, 3)
        >>> surfaces(jnp.zeros((4, 1)), jnp.zeros((1, 5))).shape
        (2, 3, 4, 5)

        ```
        """
        return self.numerator(*ts) / self.denominator(*ts)

    def _check_other(self, other: object):
        if not isinstance(other, type(self)):
            raise TypeError(
                f"{type(self).__name__} arithmetic requires another "
                f"{type(self).__name__}"
            )
        return other

    def __add__(self, other: object):
        r"""Return the exact pointwise sum.

        For $R_1=N_1/D_1$ and $R_2=N_2/D_2$,

        $$
        R_1+R_2=\frac{N_1D_2+N_2D_1}{D_1D_2}.
        $$

        Tensor Bernstein products add the degree vectors componentwise, and
        ``other`` must have the same rational tensor type. See
        :class:`_RationalTensorBernstein` for the shared arithmetic batch
        convention.
        """
        other = self._check_other(other)
        denominator = self.denominator * other.denominator
        numerator = (
            self.numerator * other.denominator + other.numerator * self.denominator
        )
        return type(self)._from_homogeneous(numerator.c, denominator.c)

    def __sub__(self, other: object):
        r"""Return the exact pointwise difference.

        For $R_1=N_1/D_1$ and $R_2=N_2/D_2$,

        $$
        R_1-R_2=\frac{N_1D_2-N_2D_1}{D_1D_2}.
        $$

        The denominator product remains positive throughout $[0,1]^d$ and
        degree vectors add componentwise. The shared arithmetic batch
        convention is documented by :class:`_RationalTensorBernstein`.
        """
        other = self._check_other(other)
        denominator = self.denominator * other.denominator
        numerator = (
            self.numerator * other.denominator - other.numerator * self.denominator
        )
        return type(self)._from_homogeneous(numerator.c, denominator.c)

    def __mul__(self, other: object):
        r"""Return the exact pointwise product.

        For $R_1=N_1/D_1$ and $R_2=N_2/D_2$,

        $$
        R_1R_2=\frac{N_1N_2}{D_1D_2}.
        $$

        Tensor Bernstein multiplication gives degree vector
        $\mathbf{n}_1+\mathbf{n}_2$ and preserves denominator positivity. The
        shared arithmetic batch convention is documented by
        :class:`_RationalTensorBernstein`.
        """
        other = self._check_other(other)
        numerator = self.numerator * other.numerator
        denominator = self.denominator * other.denominator
        return type(self)._from_homogeneous(numerator.c, denominator.c)

    def deriv(self, m: int = 1, axis: int = 0):
        r"""Return the exact partial derivative $\partial_{u_a}^mR$.

        With $a=\text{axis}$, one step applies

        $$
        \partial_{u_a}R
        =\frac{(\partial_{u_a}N)D-N(\partial_{u_a}D)}{D^2}.
        $$

        The quotient rule is repeated $m$ times. After each step, the
        numerator is degree-elevated along any shorter parameter axis so it
        shares the denominator's degree vector; this does not change the
        represented function. $m=0$ returns ``self``.

        Batch dimensions
        ----------------

        Partial differentiation acts independently on every batch item. It
        preserves all leading axes and changes only the homogeneous
        coefficient degrees:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import RationalBernstein2D
        >>> surfaces = RationalBernstein2D(
        ...     jnp.zeros((2, 3, 3, 4)),
        ...     jnp.ones((3, 4)),
        ... )
        >>> surfaces.deriv(axis=1).shape
        (2, 3)
        >>> surfaces.deriv(axis=1).h.shape
        (2, 3, 2, 5, 7)

        ```
        """
        self.numerator._axis_index(axis)
        if m < 0:
            raise ValueError("derivative order must be non-negative")
        result = self
        for _ in range(m):
            numerator = result.numerator.deriv(
                axis=axis
            ) * result.denominator - result.numerator * result.denominator.deriv(
                axis=axis
            )
            denominator = result.denominator * result.denominator
            target_degrees = tuple(int(value) for value in denominator.order)
            numerator_coefficients = numerator._elevate(numerator.c, target_degrees)
            result = type(self)._from_homogeneous(numerator_coefficients, denominator.c)
        return result

    def weight_sensitivity(self) -> "_RationalTensorBernstein":
        r"""Return sensitivities to every tensor-product control weight.

        The returned object represents $\partial R/\partial w_{\mathbf{i}}$
        for every coefficient multi-index $\mathbf{i}$, with dehomogenized controls held fixed.  Its shape
        is ``self.shape + coefficient_shape``: the added trailing batch axes
        select the differentiated coefficient.  Every parameter degree is
        doubled, because each sensitivity is

        $$
        \frac{B_{\mathbf i}^{\mathbf n}
        (c_{\mathbf i}D-N)}{D^2}.
        $$
        """
        dimensions = self.parameter_dimensions
        coefficient_shape = self.h.shape[-dimensions:]
        batch_shape = self.shape
        numerator = self.numerator
        denominator = self.denominator
        coefficient_count = math.prod(coefficient_shape)
        basis_coefficients = jnp.eye(
            coefficient_count, dtype=self.h.dtype
        ).reshape(coefficient_shape + coefficient_shape)
        basis = self.polynomial_type(basis_coefficients)
        selector_shape = coefficient_shape
        coefficient_axes = (1,) * dimensions
        difference = self.polynomial_type(
            self.values.reshape(batch_shape + selector_shape + coefficient_axes)
            * denominator.c.reshape(batch_shape + coefficient_axes + coefficient_shape)
            - numerator.c.reshape(batch_shape + coefficient_axes + coefficient_shape)
        )
        sensitivity_numerator = basis * difference
        sensitivity_denominator = denominator * denominator
        denominator_coefficients = jnp.broadcast_to(
            sensitivity_denominator.c.reshape(
                batch_shape
                + coefficient_axes
                + sensitivity_denominator.c.shape[-dimensions:]
            ),
            sensitivity_numerator.c.shape,
        )
        return type(self)._from_homogeneous(
            sensitivity_numerator.c, denominator_coefficients
        )

    def split(
        self, t: Float[jax.Array, "..."] | float = jnp.array(0.5), axis: int = 0
    ):
        r"""Split along $u_a=\tau$ and reparameterize both pieces.

        With $a=\text{axis}$ and a new coordinate $s\in[0,1]$, the first result
        represents

        $$
        R_{\mathrm{left}}(\mathbf{u}_{\neg a},s)
        =R(u_0,\ldots,\tau s,\ldots,u_{d-1}),
        $$

        while the second represents

        $$
        R_{\mathrm{right}}(\mathbf{u}_{\neg a},s)
        =R(u_0,\ldots,\tau+(1-\tau)s,\ldots,u_{d-1}).
        $$

        Homogeneous numerator and denominator tensors are subdivided by De
        Casteljau. Positive weights remain positive, and both results retain
        the input type, degree vector, and batch shape.

        Batch dimensions
        ----------------

        A scalar $\tau$ is shared by all batch items. An array with
        ``self.shape`` may select one split value per item; both pieces retain
        the original homogeneous shape:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import RationalBernstein2D
        >>> surfaces = RationalBernstein2D(
        ...     jnp.zeros((2, 2, 3)),
        ...     jnp.ones((2, 2, 3)),
        ... )
        >>> left, right = surfaces.split(jnp.array([0.25, 0.75]), axis=0)
        >>> left.h.shape, right.h.shape
        ((2, 2, 2, 3), (2, 2, 2, 3))

        ```
        """
        numerator_left, numerator_right = self.numerator.split(t=t, axis=axis)
        denominator_left, denominator_right = self.denominator.split(t=t, axis=axis)
        return (
            type(self)._from_homogeneous(numerator_left.c, denominator_left.c),
            type(self)._from_homogeneous(numerator_right.c, denominator_right.c),
        )

    def _lower_dimension(
        self,
        numerator: Float[jax.Array, "..."],
        denominator: Float[jax.Array, "..."],
    ):
        if self.parameter_dimensions == 2:
            return _from_homogeneous(numerator, denominator)
        if self.parameter_dimensions == 3:
            return RationalBernstein2D._from_homogeneous(numerator, denominator)
        if self.parameter_dimensions == 4:
            return RationalBernstein3D._from_homogeneous(numerator, denominator)
        raise RuntimeError("rational tensor functions require at least 2 dimensions")

    def slice(self, value: Float[jax.Array, "..."] | float, axis: int = 0):
        r"""Restrict to the coordinate hyperplane $u_a=v$.

        For $a=\text{axis}$ and $v=\text{value}$, the returned function is

        $$
        Q(u_0,\ldots,u_{a-1},u_{a+1},\ldots,u_{d-1})
        =R(u_0,\ldots,u_{a-1},v,u_{a+1},\ldots,u_{d-1}).
        $$

        De Casteljau evaluation removes the selected parameter-degree axis
        from both homogeneous components. The return type has one fewer
        parameter dimension: 2D becomes :class:`RationalBernstein`, 3D becomes
        :class:`RationalBernstein2D`, and 4D becomes
        :class:`RationalBernstein3D`. ``value`` broadcasts with leading batch
        axes.

        Batch dimensions
        ----------------

        The fixed value broadcasts against ``self.shape``. The result retains
        the broadcast batch axes and removes exactly one parameter-degree
        axis:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import RationalBernstein2D
        >>> surfaces = RationalBernstein2D(
        ...     jnp.zeros((2, 3, 2, 3)),
        ...     jnp.ones((2, 3, 2, 3)),
        ... )
        >>> curves = surfaces.slice(jnp.zeros((2, 1)), axis=0)
        >>> curves.shape, curves.h.shape
        ((2, 3), (2, 3, 2, 3))

        ```
        """
        numerator = self.numerator.slice(value=value, axis=axis)
        denominator = self.denominator.slice(value=value, axis=axis)
        return self._lower_dimension(numerator.c, denominator.c)

    def segment(
        self,
        start: Float[jax.Array, "..."],
        end: Float[jax.Array, "..."],
    ) -> RationalBernstein:
        r"""Restrict $R$ to the affine segment from ``start`` to ``end``.

        For endpoints $\mathbf{a}$ and $\mathbf{b}$, the returned
        :class:`RationalBernstein` represents

        $$
        Q(t)=R\bigl(\mathbf{a}+t(\mathbf{b}-\mathbf{a})\bigr),
        \qquad 0\leq t\leq1.
        $$

        The tensor numerator and denominator are each transformed exactly to
        a univariate Bernstein polynomial using the tensor-product blossom.
        Their resulting coefficients form the homogeneous representation of
        $Q$. Endpoint arrays have final length $d$ and their leading shapes
        broadcast with ``self.shape``.

        Batch dimensions
        ----------------

        ``start`` and ``end`` have shapes ``(*endpoint_batch, d)``. Their
        leading shapes broadcast with each other and with ``self.shape``.
        Every tensor degree axis is replaced by one univariate axis of degree
        $\sum_a n_a$:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import RationalBernstein2D
        >>> surfaces = RationalBernstein2D(
        ...     jnp.zeros((2, 3, 2, 3)),
        ...     jnp.ones((2, 3, 2, 3)),
        ... )
        >>> curves = surfaces.segment(
        ...     jnp.zeros((2, 1, 2)),
        ...     jnp.ones((1, 3, 2)),
        ... )
        >>> curves.shape, curves.h.shape
        ((2, 3), (2, 3, 2, 4))

        ```
        """
        numerator = self.numerator.segment(start, end)
        denominator = self.denominator.segment(start, end)
        return _from_homogeneous(numerator.c, denominator.c)


class RationalBernstein2D(_RationalTensorBernstein):
    r"""Represent a positive-weight scalar rational Bernstein function on $[0,1]^2$.

    ``values`` and ``weights`` have shape
    ``(*batch, n_x + 1, n_y + 1)`` and define

    $$
    R(x,y)=
    \frac{\sum_{i=0}^{n_x}\sum_{j=0}^{n_y}
    w_{ij}c_{ij}B_i^{n_x}(x)B_j^{n_y}(y)}
    {\sum_{i=0}^{n_x}\sum_{j=0}^{n_y}
    w_{ij}B_i^{n_x}(x)B_j^{n_y}(y)}.
    $$

    The inherited :attr:`order` is $[n_x,n_y]$. See
    :class:`_RationalTensorBernstein` for homogeneous layout, constructor and
    arithmetic batch broadcasting, evaluation axes, and operation semantics.
    """

    parameter_dimensions: ClassVar[int] = 2
    polynomial_type: ClassVar[type] = Bernstein2D

    def __init__(self, values: Float[jax.Array, "*batch coefficient"], weights: Float[jax.Array, "*batch coefficient"]):
        """Initialize 2D control values and strictly positive weights."""
        super().__init__(values, weights)


class RationalBernstein3D(_RationalTensorBernstein):
    r"""Represent a positive-weight scalar rational Bernstein function on $[0,1]^3$.

    The coefficient arrays have shape
    ``(*batch, n_x + 1, n_y + 1, n_z + 1)``. With
    $\mathbf{u}=(x,y,z)$, the function is

    $$
    R(\mathbf{u})=
    \frac{\sum_{i,j,k}w_{ijk}c_{ijk}
    B_i^{n_x}(x)B_j^{n_y}(y)B_k^{n_z}(z)}
    {\sum_{i,j,k}w_{ijk}
    B_i^{n_x}(x)B_j^{n_y}(y)B_k^{n_z}(z)}.
    $$

    The inherited :attr:`order` is $[n_x,n_y,n_z]$. See
    :class:`_RationalTensorBernstein` for the shared batch and arithmetic
    conventions.
    """

    parameter_dimensions: ClassVar[int] = 3
    polynomial_type: ClassVar[type] = Bernstein3D

    def __init__(self, values: Float[jax.Array, "*batch coefficient"], weights: Float[jax.Array, "*batch coefficient"]):
        """Initialize 3D control values and strictly positive weights."""
        super().__init__(values, weights)


class RationalBernstein4D(_RationalTensorBernstein):
    r"""Represent a positive-weight scalar rational Bernstein function on $[0,1]^4$.

    The coefficient arrays have shape
    ``(*batch, n_0 + 1, n_1 + 1, n_2 + 1, n_3 + 1)`` and define

    $$
    R(\mathbf{u})=
    \frac{\sum_{\mathbf{i}}w_{\mathbf{i}}c_{\mathbf{i}}
    \prod_{a=0}^3B_{i_a}^{n_a}(u_a)}
    {\sum_{\mathbf{i}}w_{\mathbf{i}}
    \prod_{a=0}^3B_{i_a}^{n_a}(u_a)}.
    $$

    The inherited :attr:`order` is $[n_0,n_1,n_2,n_3]$. See
    :class:`_RationalTensorBernstein` for the shared batch and arithmetic
    conventions.
    """

    parameter_dimensions: ClassVar[int] = 4
    polynomial_type: ClassVar[type] = Bernstein4D

    def __init__(self, values: Float[jax.Array, "*batch coefficient"], weights: Float[jax.Array, "*batch coefficient"]):
        """Initialize 4D control values and strictly positive weights."""
        super().__init__(values, weights)


def _evaluate_rational_tensor(
    h: Float[jax.Array, "..."], point: Float[jax.Array, "..."]
) -> Float[jax.Array, "..."]:
    numerator = _evaluate_tensor_coefficients(h[0], point)
    denominator = _evaluate_tensor_coefficients(h[1], point)
    return numerator / denominator


def _split_rational_tensor(
    h: Float[jax.Array, "..."], axis: Int[jax.Array, ""]
) -> tuple[Float[jax.Array, "..."], Float[jax.Array, "..."]]:
    numerator_left, numerator_right = _split_tensor_coefficients(h[0], axis)
    denominator_left, denominator_right = _split_tensor_coefficients(h[1], axis)
    return (
        jnp.stack((numerator_left, denominator_left)),
        jnp.stack((numerator_right, denominator_right)),
    )


@jax.custom_jvp
def _minimize(
    h: Float[jax.Array, "..."], max_steps: int = 200, eps: float = 1e-6
) -> tuple[Float[jax.Array, ""], Float[jax.Array, "dim"]]:
    """Globally minimize one unbatched positive-weight rational tensor."""
    dimensions = h.ndim - 1
    capacity = max_steps + 1
    corners = jnp.asarray(list(itertools.product((0, 1), repeat=dimensions)))
    samples = jnp.concatenate([corners, jnp.full((1, dimensions), 0.5, dtype=h.dtype)])
    initial_values = jax.vmap(lambda point: _evaluate_rational_tensor(h, point))(
        samples
    )
    initial_index = jnp.argmin(initial_values)
    lower = jnp.zeros((capacity, dimensions), dtype=h.dtype)
    upper = jnp.ones((capacity, dimensions), dtype=h.dtype)
    control = jnp.zeros((capacity,) + h.shape, dtype=h.dtype).at[0].set(h)
    values = h[0] / h[1]
    bounds = jnp.full(capacity, jnp.inf, dtype=h.dtype).at[0].set(jnp.min(values))
    state = (
        lower,
        upper,
        control,
        bounds,
        initial_values[initial_index],
        samples[initial_index],
        0,
    )

    def condition(current):
        _, _, _, current_bounds, value, _, step = current
        return (step < max_steps) & ((value - jnp.min(current_bounds)) > eps)

    def body(current):
        lower, upper, control, bounds, value, point, step = current
        index = jnp.argmin(bounds)
        lo, hi, current_h = lower[index], upper[index], control[index]
        axis = jnp.argmax(hi - lo)
        left, right = _split_rational_tensor(current_h, axis)
        midpoint = 0.5 * (lo + hi)
        candidate_points = lo + samples * (hi - lo)
        candidate_values = jax.vmap(
            lambda candidate: _evaluate_rational_tensor(current_h, candidate)
        )(samples)
        candidate_index = jnp.argmin(candidate_values)
        candidate_value = candidate_values[candidate_index]
        replace = candidate_value < value
        value = jnp.where(replace, candidate_value, value)
        point = jnp.where(replace, candidate_points[candidate_index], point)
        next_index = step + 1
        lower = (
            lower.at[index].set(lo).at[next_index].set(lo.at[axis].set(midpoint[axis]))
        )
        upper = (
            upper.at[index].set(hi.at[axis].set(midpoint[axis])).at[next_index].set(hi)
        )
        control = control.at[index].set(left).at[next_index].set(right)
        left_values = left[0] / left[1]
        right_values = right[0] / right[1]
        bounds = (
            bounds.at[index]
            .set(jnp.min(left_values))
            .at[next_index]
            .set(jnp.min(right_values))
        )
        return lower, upper, control, bounds, value, point, step + 1

    result = jax.lax.while_loop(condition, body, state)
    return result[4], result[5]


@_minimize.defjvp
def _minimize_jvp(
    primals: tuple[Float[jax.Array, "..."], int, float],
    tangents: tuple[Float[jax.Array, "..."], int, float],
) -> tuple[tuple[Float[jax.Array, ""], Float[jax.Array, "dim"]], tuple[Float[jax.Array, ""], Float[jax.Array, "dim"]]]:
    h, max_steps, eps = primals
    tangent_h, _, _ = tangents
    primal = _minimize(h, max_steps=max_steps, eps=eps)
    point = primal[1]
    tangent_value = jax.jvp(
        lambda homogeneous: _evaluate_rational_tensor(homogeneous, point),
        (h,),
        (tangent_h,),
    )[1]

    def gradient(homogeneous):
        return jax.grad(
            lambda parameters: _evaluate_rational_tensor(homogeneous, parameters)
        )(point)

    tangent_gradient = jax.jvp(gradient, (h,), (tangent_h,))[1]
    hessian = jax.hessian(lambda parameters: _evaluate_rational_tensor(h, parameters))(
        point
    )
    tangent_point = jax.lax.cond(
        jnp.any((point == 0.0) | (point == 1.0)),
        lambda _: jnp.zeros_like(point),
        lambda _: -jnp.linalg.solve(hessian, tangent_gradient),
        operand=None,
    )
    return primal, (tangent_value, tangent_point)
