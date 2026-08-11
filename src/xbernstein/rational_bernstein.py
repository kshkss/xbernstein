r"""One-dimensional scalar rational Bernstein functions on $[0,1]$."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float

from .bernstein import Bernstein, _elevate


def _from_homogeneous(
    numerator: jax.Array, denominator: jax.Array
) -> "RationalBernstein":
    result = object.__new__(RationalBernstein)
    object.__setattr__(result, "h", jnp.stack((numerator, denominator), axis=-2))
    return result


def _split_homogeneous(h: jax.Array, t: jax.Array | float):
    t = jnp.asarray(t, dtype=h.dtype)
    t = jnp.broadcast_to(t, h.shape[:-2])
    weight = t[..., None, None]
    work = h
    left = [work[..., 0]]
    right = [work[..., -1]]
    for _ in range(h.shape[-1] - 1):
        work = (1.0 - weight) * work[..., :-1] + weight * work[..., 1:]
        left.append(work[..., 0])
        right.append(work[..., -1])
    return jnp.stack(left, axis=-1), jnp.stack(right[::-1], axis=-1)


def _evaluate_homogeneous(h: jax.Array, t: jax.Array):
    numerator = Bernstein(h[..., 0, :])(t)
    denominator = Bernstein(h[..., 1, :])(t)
    return numerator / denominator


class RationalBernstein(eqx.Module):
    r"""Represent a scalar rational Bernstein function with positive weights.

    For degree $n$, ``RationalBernstein(values, weights)`` represents

    $$
    R(t)=\frac{N(t)}{D(t)}
    =\frac{\displaystyle\sum_{i=0}^n w_i c_i B_i^n(t)}
           {\displaystyle\sum_{i=0}^n w_i B_i^n(t)},
    \qquad
    B_i^n(t)=\binom{n}{i}t^i(1-t)^{n-i}.
    $$

    Here $c_i$ is a control value and $w_i>0$ is its weight. Since the
    Bernstein basis is nonnegative and sums to one on $[0,1]$,

    $$
    D(t)=\sum_{i=0}^n w_i B_i^n(t)>0,
    $$

    so $R$ has no pole on its parameter domain.

    Homogeneous representation
    --------------------------

    The stored array ``h`` contains the numerator and denominator Bernstein
    coefficients:

    $$
    h_{0,i}=w_i c_i,\qquad h_{1,i}=w_i.
    $$

    Thus ``h[..., 0, :] = weights * values`` and
    ``h[..., 1, :] = weights``. The final axis is the Bernstein index $i$ and
    the preceding length-2 axis selects $(N,D)$.

    Batch dimensions
    ----------------

    ``values`` and ``weights`` have shapes
    ``(*value_batch, n + 1)`` and ``(*weight_batch, n + 1)``. Their leading
    shapes broadcast to ``*batch``; the final axis lengths must agree.

    The homogeneous-component and coefficient axes are structural, not batch
    axes. Consequently, raw arrays with shapes ``(2, 1, 3)`` and ``(1, 4, 3)``
    construct a rational family with batch shape ``(2, 4)`` and degree $2$:

    ```python
    >>> import jax.numpy as jnp
    >>> from xbernstein import RationalBernstein
    >>> curves = RationalBernstein(
    ...     jnp.zeros((2, 1, 3)),
    ...     jnp.ones((1, 4, 3)),
    ... )
    >>> curves.shape, curves.h.shape
    ((2, 4), (2, 4, 2, 3))

    ```

    Arithmetic broadcasts only leading batch axes. The coefficient axis is
    aligned by polynomial degree instead of ordinary array broadcasting. For
    example, a degree-2 batch and a scalar-batch degree-1 function can be
    added even though their raw coefficient shapes do not align:

    ```python
    >>> left = RationalBernstein(jnp.zeros((2, 3)), jnp.ones(3))
    >>> right = RationalBernstein(jnp.zeros(2), jnp.ones(2))
    >>> (left + right).shape, (left + right).order
    ((2,), 3)

    ```

    The operators ``+``, ``-``, and ``*`` all use right-aligned broadcasting
    for leading axes. Their exact common-denominator formulas multiply the
    operands, so the result degree is the sum of input degrees. Incompatible
    leading shapes raise ``ValueError``:

    ```python
    >>> first = RationalBernstein(jnp.zeros((2, 3)), jnp.ones((2, 3)))
    >>> second = RationalBernstein(jnp.zeros((3, 3)), jnp.ones((3, 3)))
    >>> try:
    ...     first + second
    ... except ValueError as error:
    ...     type(error).__name__
    'ValueError'

    ```

    Parameter arrays do not pair with batch axes. Their shape is appended
    after ``*batch``:

    ```python
    >>> curves(jnp.linspace(0.0, 1.0, 5)).shape
    (2, 4, 5)

    ```
    """

    h: Float[jax.Array, "*batch 2 order"]

    def __init__(self, values, weights):
        r"""Initialize control values $c_i$ and strictly positive weights $w_i$.

        The inputs have shapes ``(*value_batch, n + 1)`` and
        ``(*weight_batch, n + 1)``. Their leading axes broadcast, while their
        final coefficient-axis lengths must match. The constructor stores
        $(w_i c_i,w_i)$ in homogeneous form and rejects every $w_i\leq0$.

        See :class:`RationalBernstein` for examples of constructor batch
        broadcasting and the resulting homogeneous shape.
        """
        values = jnp.asarray(values)
        weights = jnp.asarray(weights)
        if values.ndim == 0 or weights.ndim == 0:
            raise ValueError("values and weights must have a coefficient axis")
        if values.shape[-1] != weights.shape[-1]:
            raise ValueError(
                "values and weights must have the same coefficient-axis length"
            )

        dtype = jnp.result_type(values, weights, jnp.float32)
        values = values.astype(dtype)
        weights = weights.astype(dtype)
        batch_shape = jnp.broadcast_shapes(values.shape[:-1], weights.shape[:-1])
        coefficient_shape = (values.shape[-1],)
        values = jnp.broadcast_to(values, batch_shape + coefficient_shape)
        weights = jnp.broadcast_to(weights, batch_shape + coefficient_shape)
        weights = eqx.error_if(
            weights,
            jnp.any(weights <= 0.0),
            "RationalBernstein weights must be strictly positive",
        )
        self.h = jnp.stack((weights * values, weights), axis=-2)

    @property
    def values(self) -> jax.Array:
        r"""Return the dehomogenized control values $c_i$.

        The homogeneous components satisfy

        $$
        c_i=\frac{h_{0,i}}{h_{1,i}}.
        $$

        The returned shape is ``(*batch, n + 1)``.
        """
        return self.h[..., 0, :] / self.h[..., 1, :]

    @property
    def weights(self) -> jax.Array:
        r"""Return the strictly positive rational weights $w_i=h_{1,i}$.

        The returned shape is ``(*batch, n + 1)`` and every entry is positive,
        preserving $D(t)>0$ on $[0,1]$.
        """
        return self.h[..., 1, :]

    @property
    def c(self) -> jax.Array:
        r"""Return the control values $c_i$ as an alias of :attr:`values`.

        This name matches the coefficient notation used by :class:`Bernstein`;
        for a rational function these are dehomogenized values, not the
        numerator coefficients $w_i c_i$.
        """
        return self.values

    @property
    def w(self) -> jax.Array:
        r"""Return the positive weights $w_i$ as an alias of :attr:`weights`."""
        return self.weights

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Return the leading batch shape ``*batch``.

        The homogeneous array has shape ``(*batch, 2, n + 1)``; this property
        removes both the homogeneous-component and Bernstein-index axes.
        """
        return self.h.shape[:-2]

    @property
    def order(self) -> int:
        r"""Return the common numerator/denominator degree $n$.

        The final homogeneous axis stores indices $0,\ldots,n$, so
        ``h.shape[-1] == order + 1``.
        """
        return self.h.shape[-1] - 1

    @property
    def dtype(self) -> str:
        r"""Return the scalar dtype shared by $w_i c_i$ and $w_i$."""
        return str(self.h.dtype)

    @property
    def numerator(self) -> Bernstein:
        r"""Return the polynomial numerator $N$ as a :class:`Bernstein`.

        Its coefficients are

        $$
        n_i=w_i c_i=h_{0,i},
        \qquad
        N(t)=\sum_{i=0}^n n_i B_i^n(t).
        $$
        """
        return Bernstein(self.h[..., 0, :])

    @property
    def denominator(self) -> Bernstein:
        r"""Return the positive polynomial denominator $D$ as a :class:`Bernstein`.

        Its coefficients are

        $$
        d_i=w_i=h_{1,i},
        \qquad
        D(t)=\sum_{i=0}^n d_i B_i^n(t)>0
        \quad (0\leq t\leq1).
        $$
        """
        return Bernstein(self.h[..., 1, :])

    def __call__(self, t):
        r"""Evaluate $R(t)=N(t)/D(t)$.

        Both $N$ and $D$ are evaluated by the Bernstein De Casteljau
        recurrence, after which their values are divided. A scalar ``t`` is
        shared by every batch item. If ``t`` has shape ``*points``, the result
        has shape ``(*batch, *points)``.

        Batch dimensions
        ----------------

        Parameter axes are appended after all leading coefficient axes:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import RationalBernstein
        >>> curves = RationalBernstein(jnp.zeros((2, 3, 4)), jnp.ones(4))
        >>> curves(0.25).shape
        (2, 3)
        >>> curves(jnp.zeros((5, 1))).shape
        (2, 3, 5, 1)

        ```
        """
        return _evaluate_homogeneous(self.h, t)

    def _check_other(self, other: object) -> "RationalBernstein":
        if not isinstance(other, type(self)):
            raise TypeError(
                f"{type(self).__name__} arithmetic requires another "
                f"{type(self).__name__}"
            )
        return other

    def __add__(self, other: object) -> "RationalBernstein":
        r"""Return the exact pointwise sum of two rational functions.

        If ``self`` is $N_1/D_1$ and ``other`` is $N_2/D_2$, the result is

        $$
        \frac{N_1}{D_1}+\frac{N_2}{D_2}
        =\frac{N_1D_2+N_2D_1}{D_1D_2}.
        $$

        Bernstein polynomial products make this identity exact. For input
        degrees $n_1,n_2$, the returned degree is $n_1+n_2$. Only another
        :class:`RationalBernstein` is accepted. See
        :class:`RationalBernstein` for the shared arithmetic batch convention.
        """
        other = self._check_other(other)
        denominator = self.denominator * other.denominator
        numerator = self.numerator * other.denominator
        numerator = numerator + other.numerator * self.denominator
        return _from_homogeneous(numerator.c, denominator.c)

    def __sub__(self, other: object) -> "RationalBernstein":
        r"""Return the exact pointwise difference of two rational functions.

        For $R_1=N_1/D_1$ and $R_2=N_2/D_2$,

        $$
        R_1-R_2=\frac{N_1D_2-N_2D_1}{D_1D_2}.
        $$

        The positive denominator product preserves the no-pole invariant and
        degrees add. The shared arithmetic batch convention is documented by
        :class:`RationalBernstein`.
        """
        other = self._check_other(other)
        denominator = self.denominator * other.denominator
        numerator = self.numerator * other.denominator
        numerator = numerator - other.numerator * self.denominator
        return _from_homogeneous(numerator.c, denominator.c)

    def __mul__(self, other: object) -> "RationalBernstein":
        r"""Return the exact pointwise product of two rational functions.

        For $R_1=N_1/D_1$ and $R_2=N_2/D_2$,

        $$
        R_1R_2=\frac{N_1N_2}{D_1D_2}.
        $$

        The returned degree is the sum of the input degrees. Since
        $D_1,D_2>0$ on $[0,1]$, their product is also positive. See
        :class:`RationalBernstein` for the shared arithmetic batch convention.
        """
        other = self._check_other(other)
        numerator = self.numerator * other.numerator
        denominator = self.denominator * other.denominator
        return _from_homogeneous(numerator.c, denominator.c)

    def deriv(self, m: int = 1) -> "RationalBernstein":
        r"""Return the exact $m$-th derivative as a rational Bernstein function.

        One derivative applies the quotient rule

        $$
        R'(t)
        =\frac{N'(t)D(t)-N(t)D'(t)}{D(t)^2}.
        $$

        The operation is repeated ``m`` times. After every step the numerator
        is degree-elevated, if necessary, to share the denominator's Bernstein
        degree; degree elevation changes the representation but not the
        represented function. ``m=0`` returns ``self`` and a negative ``m`` is
        invalid.

        Batch dimensions
        ----------------

        Differentiation acts independently on every batch item and changes
        only the final coefficient axis:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import RationalBernstein
        >>> curves = RationalBernstein(jnp.zeros((2, 3, 4)), jnp.ones(4))
        >>> curves.deriv().shape
        (2, 3)
        >>> curves.deriv().h.shape[:2]
        (2, 3)

        ```
        """
        if m < 0:
            raise ValueError("derivative order must be non-negative")
        result = self
        for _ in range(m):
            numerator = (
                result.numerator.deriv() * result.denominator
                - result.numerator * result.denominator.deriv()
            )
            denominator = result.denominator * result.denominator
            numerator_coefficients = _elevate(numerator.c, denominator.order)
            result = _from_homogeneous(numerator_coefficients, denominator.c)
        return result

    def split(
        self, t: jax.Array | float = jnp.array(0.5)
    ) -> tuple["RationalBernstein", "RationalBernstein"]:
        r"""Split at $t=\tau$ and reparameterize both pieces to $[0,1]$.

        For the original function $R$, the returned ``left`` and ``right``
        satisfy

        $$
        R_{\mathrm{left}}(s)=R(\tau s),\qquad
        R_{\mathrm{right}}(s)=R\bigl(\tau+(1-\tau)s\bigr),
        \qquad 0\leq s\leq1.
        $$

        De Casteljau subdivision is applied to both homogeneous components.
        It uses convex combinations, so positive denominator coefficients
        remain positive. A batched ``t`` broadcasts over ``self.shape``.

        Batch dimensions
        ----------------

        A scalar split parameter is shared by all batch items. A parameter
        with shape ``self.shape`` selects a different split for each item;
        both results retain the original homogeneous shape:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import RationalBernstein
        >>> curves = RationalBernstein(jnp.zeros((2, 3)), jnp.ones((2, 3)))
        >>> left, right = curves.split(jnp.array([0.25, 0.75]))
        >>> left.h.shape, right.h.shape
        ((2, 2, 3), (2, 2, 3))

        ```
        """
        left, right = _split_homogeneous(self.h, t)
        return (
            _from_homogeneous(left[..., 0, :], left[..., 1, :]),
            _from_homogeneous(right[..., 0, :], right[..., 1, :]),
        )


@jax.custom_jvp
def _minimize(h: jax.Array, max_steps: int = 200, eps: float = 1e-6):
    """Approximate the minimum of one unbatched positive-weight curve."""
    max_cap = max_steps + 1
    degree = h.shape[-1] - 1
    values = h[0] / h[1]

    u_buffer = jnp.zeros(max_cap).at[0].set(0.0)
    v_buffer = jnp.zeros(max_cap).at[0].set(1.0)
    h_buffer = jnp.zeros((max_cap, 2, degree + 1)).at[0].set(h)
    lower_buffer = jnp.full(max_cap, jnp.inf).at[0].set(jnp.min(values))
    upper = jnp.minimum(values[0], values[-1])
    best_x = jnp.where(values[0] < values[-1], 0.0, 1.0)
    initial = (u_buffer, v_buffer, h_buffer, lower_buffer, upper, best_x, 0)

    def condition(state):
        _, _, _, lower, incumbent, _, step = state
        return (step < max_steps) & ((incumbent - jnp.min(lower)) > eps)

    def body(state):
        u_buffer, v_buffer, h_buffer, lower, incumbent, best_x, step = state
        index = jnp.argmin(lower)
        u = u_buffer[index]
        v = v_buffer[index]
        current = h_buffer[index]
        midpoint = 0.5 * (u + v)
        left, right = _split_homogeneous(current, jnp.array(0.5, dtype=current.dtype))
        left_values = left[0] / left[1]
        right_values = right[0] / right[1]

        samples = jnp.array([left_values[0], left_values[-1], right_values[-1]])
        sample_points = jnp.array([u, midpoint, v])
        sample_index = jnp.argmin(samples)
        new_upper = jnp.minimum(incumbent, samples[sample_index])
        new_best_x = jnp.where(
            samples[sample_index] < incumbent,
            sample_points[sample_index],
            best_x,
        )

        next_index = step + 1
        u_buffer = u_buffer.at[index].set(u).at[next_index].set(midpoint)
        v_buffer = v_buffer.at[index].set(midpoint).at[next_index].set(v)
        h_buffer = h_buffer.at[index].set(left).at[next_index].set(right)
        lower = (
            lower.at[index]
            .set(jnp.min(left_values))
            .at[next_index]
            .set(jnp.min(right_values))
        )
        return (
            u_buffer,
            v_buffer,
            h_buffer,
            lower,
            new_upper,
            new_best_x,
            step + 1,
        )

    final = jax.lax.while_loop(condition, body, initial)
    return final[4], final[5]


@_minimize.defjvp
def _minimize_jvp(primals, tangents):
    h, max_steps, eps = primals
    tangent_h, _, _ = tangents
    primal = _minimize(h, max_steps=max_steps, eps=eps)
    x_star = primal[1]

    tangent_f = jax.jvp(
        lambda homogeneous: _evaluate_homogeneous(homogeneous, x_star),
        (h,),
        (tangent_h,),
    )[1]

    def first_derivative(homogeneous):
        return jax.grad(
            lambda parameter: _evaluate_homogeneous(homogeneous, parameter)
        )(x_star)

    tangent_first = jax.jvp(first_derivative, (h,), (tangent_h,))[1]
    second = jax.grad(
        lambda parameter: jax.grad(lambda inner: _evaluate_homogeneous(h, inner))(
            parameter
        )
    )(x_star)
    tangent_x = jax.lax.cond(
        jnp.equal(x_star, 0.0) | jnp.equal(x_star, 1.0),
        lambda _: jnp.zeros_like(x_star),
        lambda _: -tangent_first / second,
        operand=None,
    )
    return primal, (tangent_f, tangent_x)
