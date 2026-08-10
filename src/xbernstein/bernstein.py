r"""One-dimensional Bernstein polynomials on the unit interval.

For degree $n$, the Bernstein basis is

$$
B_i^n(t) = \binom{n}{i} t^i (1-t)^{n-i},
\qquad 0 \leq i \leq n,\quad t \in [0, 1].
$$

Throughout this module, $\mathbf{c} = (c_0, \ldots, c_n)$ denotes a
coefficient vector and

$$
p(t) = \sum_{i=0}^{n} c_i B_i^n(t)
$$

denotes its polynomial. The final array axis stores $i$; all leading axes are
independent batch or value axes.

Batch dimensions
----------------

An array of coefficients with shape ``(*batch, n + 1)`` represents one
degree-$n$ polynomial for every leading index. Evaluating at a scalar preserves
``*batch``; evaluating at parameters with shape ``*points`` returns
``(*batch, *points)``.

```python
>>> import jax.numpy as jnp
>>> from xbernstein import Bernstein
>>> coefficients = jnp.array([[0.0, 1.0], [2.0, 4.0]])
>>> curves = Bernstein(coefficients)  # Two degree-1 polynomials.
>>> curves.shape
(2,)
>>> curves(0.25).shape  # One shared parameter for both curves.
(2,)
>>> curves(jnp.array([0.0, 0.5, 1.0])).shape
(2, 3)
>>> grid = Bernstein(jnp.zeros((2, 3, 2)))
>>> grid(jnp.zeros((4, 1))).shape
(2, 3, 4, 1)
```
"""

import jax
import jax.numpy as jnp
import jax.scipy.special as jss
from jaxtyping import Float, jaxtyped
from beartype import beartype
import equinox as eqx
from typing import Self, NamedTuple


def _elevate(c: jax.Array, target_n: int) -> jax.Array:
    r"""Represent $p$ in the Bernstein basis of degree $N=$ ``target_n``.

    If $p$ initially has degree $n$, degree elevation preserves the function
    while replacing its coefficient vector. One step from degree $r$ to
    $r+1$ uses

    $$
    \tilde{c}_0 = c_0,\qquad
    \tilde{c}_i = \frac{i}{r+1}c_{i-1}
        + \left(1-\frac{i}{r+1}\right)c_i,\qquad
    \tilde{c}_{r+1} = c_r.
    $$

    The recurrence is applied until degree $N$ is reached. The final axis is
    the Bernstein index $i$; leading axes are preserved unchanged.
    """
    w = c
    n = w.shape[-1] - 1

    for new_n in range(n + 1, target_n + 1):
        i = jnp.arange(1, new_n, dtype=w.dtype)
        alpha = (i / new_n).reshape((1,) * (w.ndim - 1) + (new_n - 1,))
        mid = alpha * w[..., :-1] + (1.0 - alpha) * w[..., 1:]
        w = jnp.concatenate([w[..., :1], mid, w[..., -1:]], axis=-1)

    return w


class Bernstein(eqx.Module):
    r"""Represent $p(t)=\sum_{i=0}^n c_i B_i^n(t)$ on $[0,1]$.

    The coefficient array ``c`` has shape ``(*batch, n + 1)``. Its final axis
    stores the Bernstein index $i$, while every leading axis represents an
    independent polynomial or a vector-valued coefficient component. All
    operations act on the polynomial parameter $t$ and broadcast only those
    leading axes.

    Batch dimensions
    ----------------

    Leading axes are never Bernstein axes: each leading index selects an
    independent polynomial with the same degree. The final axis is always the
    Bernstein coefficient axis, so arithmetic broadcasts only ``*batch`` and
    aligns degrees separately.

    The following raw arrays cannot be added by JAX because their final axes
    are treated as ordinary array axes:

    ```python
    >>> import jax.numpy as jnp
    >>> try:
    ...     jnp.zeros((2, 3)) + jnp.zeros((2,))
    ... except ValueError as error:
    ...     type(error).__name__
    'ValueError'
    ```

    As Bernstein coefficients, ``(2, 3)`` means batch shape ``(2,)`` and
    degree $2$, whereas ``(2,)`` means no batch axes and degree $1$.
    Therefore the second polynomial broadcasts to both batch entries and is
    degree-elevated before arithmetic:

    ```python
    >>> curves = Bernstein(jnp.array([[0.0, 1.0], [2.0, 4.0]]))
    >>> line = Bernstein(jnp.array([0.0, 1.0]))
    >>> (curves + line).c.shape
    (2, 2)
    ```

    Right-aligned broadcasting applies to every leading batch axis. The raw
    coefficient arrays below have incompatible final axes, but the Bernstein
    objects interpret those axes as degree $2$ and degree $1$:

    ```python
    >>> left = Bernstein(jnp.zeros((2, 1, 3)))   # Batch shape (2, 1).
    >>> right = Bernstein(jnp.zeros((1, 3, 2)))  # Batch shape (1, 3).
    >>> (left + right).c.shape
    (2, 3, 3)
    ```

    This applies equally to ``+``, ``-``, and ``*``. A product increases the
    degree after leading-axis broadcasting.

    Parameter arrays are not paired with batch axes. They append evaluation
    axes after ``*batch``:

    ```python
    >>> quadratic = Bernstein(jnp.zeros((2, 3)))
    >>> quadratic(jnp.array([0.0, 0.5])).shape
    (2, 2)
    >>> Bernstein(jnp.zeros((2, 3, 2)))(jnp.zeros((4, 1))).shape
    (2, 3, 4, 1)
    ```

    In contrast, two non-singleton leading batch shapes must themselves be
    broadcast-compatible:

    ```python
    >>> try:
    ...     Bernstein(jnp.zeros((2, 3))) + Bernstein(jnp.zeros((3, 2)))
    ... except ValueError as error:
    ...     type(error).__name__
    'ValueError'
    ```
    """

    c: Float[jax.Array, "*batch order"]  # 制御点

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the coefficient shape excluding the Bernstein-index axis $i$."""
        return self.c.shape[:-1]

    @property
    def order(self) -> int:
        r"""Return the polynomial degree $n$.

        The final coefficient axis stores $c_0,\ldots,c_n$, so its length
        satisfies

        $$
        \text{``c.shape[-1]``} = n + 1 = \text{``order``} + 1.
        $$
        """
        return self.c.shape[-1] - 1

    @property
    def dtype(self) -> str:
        """Return the scalar field used for the coefficients $c_i$."""
        return str(self.c.dtype)

    def __add__(self, other: Self) -> Self:
        r"""Return the polynomial sum $p+q$.

        If $p$ and $q$ have different degrees, both coefficient vectors are
        elevated to degree $\max(n,m)$ before their corresponding coefficients
        are added. Thus the returned Bernstein representation has the same
        parameter domain and function value as $p(t)+q(t)$.

        See :class:`Bernstein` for the common arithmetic batch-broadcast
        convention and executable examples.
        """
        c1 = self.c
        c2 = other.c
        target_n = max(c1.shape[-1], c2.shape[-1]) - 1
        _ = jnp.broadcast_shapes(c1.shape[:-1], c2.shape[:-1])
        return type(self)(_elevate(c1, target_n) + _elevate(c2, target_n))

    def __sub__(self, other: Self) -> Self:
        r"""Return the polynomial difference $p-q$.

        If $p$ and $q$ have different degrees, both coefficient vectors are
        elevated to degree $\max(n,m)$ before their corresponding coefficients
        are subtracted.

        See :class:`Bernstein` for the common arithmetic batch-broadcast
        convention and executable examples.
        """
        c1 = self.c
        c2 = other.c
        target_n = max(c1.shape[-1], c2.shape[-1]) - 1
        _ = jnp.broadcast_shapes(c1.shape[:-1], c2.shape[:-1])
        return type(self)(_elevate(c1, target_n) - _elevate(c2, target_n))

    def __mul__(self, other: Self) -> Self:
        r"""Return the pointwise product $r(t)=p(t)q(t)$.

        For degree-$n$ coefficients $c_i$ and degree-$m$ coefficients $d_j$,
        the product is represented at degree $n+m$:

        $$
        r_k =
        \sum_{\substack{0\leq i\leq n\\0\leq j\leq m\\i+j=k}}
        c_i d_j
        \frac{\binom{n}{i}\binom{m}{j}}{\binom{n+m}{k}}.
        $$

        This follows from the product identity for two Bernstein basis
        functions and is applied independently to every broadcast batch item.

        See :class:`Bernstein` for the common arithmetic batch-broadcast
        convention and executable examples.
        """
        c1 = self.c
        c2 = other.c
        n = c1.shape[-1] - 1
        m = c2.shape[-1] - 1
        batch_shape = jnp.broadcast_shapes(c1.shape[:-1], c2.shape[:-1])
        c1 = jnp.broadcast_to(c1, batch_shape + (n + 1,))
        c2 = jnp.broadcast_to(c2, batch_shape + (m + 1,))

        i, j = jnp.meshgrid(jnp.arange(n + 1), jnp.arange(m + 1), indexing="ij")
        scale = jss.comb(n, i) * jss.comb(m, j) / jss.comb(n + m, i + j)
        products = c1[..., :, None] * c2[..., None, :] * scale
        indices = jnp.arange(n + 1)[:, None] + jnp.arange(m + 1)[None, :]

        results = jnp.zeros(batch_shape + (n + m + 1,), dtype=c1.dtype)
        results = results.at[..., indices].add(products)

        return type(self)(results)

    def deriv(self, m=1) -> Self:
        r"""Return $\frac{d^m p}{dt^m}$ in the Bernstein basis.

        A single derivative maps a degree-$n$ coefficient vector to the
        degree-$n-1$ vector

        $$
        d_i = n(c_{i+1}-c_i),\qquad 0\leq i<n.
        $$

        The method repeats this finite-difference transformation $m$ times.
        When $m>n$, the derivative is the identically zero degree-$0$
        polynomial.

        Batch dimensions
        ----------------

        Differentiation leaves leading axes unchanged and shortens only the
        final coefficient axis:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import Bernstein
        >>> Bernstein(jnp.zeros((2, 4))).deriv(m=2).c.shape
        (2, 2)
        ```
        """
        c = self.c
        n = c.shape[-1] - 1

        if m > n:
            c = jnp.zeros(c.shape[:-1] + (1,), dtype=c.dtype)
        else:
            for degree in range(n, n - m, -1):
                c = degree * (c[..., 1:] - c[..., :-1])
        return type(self)(c)

    def int(self, k=0.0) -> Self:
        r"""Return an antiderivative $P$ such that $P'(t)=p(t)$ and $P(0)=k$.

        If $p$ has degree $n$, $P$ has degree $n+1$ with coefficients

        $$
        C_0=k,\qquad
        C_{i+1}=k+\frac{1}{n+1}\sum_{j=0}^{i}c_j,
        \qquad 0\leq i\leq n.
        $$

        The constant $k$ may broadcast over the leading coefficient axes.

        Batch dimensions
        ----------------

        Integration preserves leading axes and adds one control point on the
        final axis. A batched integration constant supplies one value per
        polynomial:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import Bernstein
        >>> Bernstein(jnp.zeros((2, 3))).int(k=jnp.array([1.0, 2.0])).c.shape
        (2, 4)
        ```
        """
        c = self.c
        n = c.shape[-1] - 1
        c_new = jnp.concatenate(
            [
                jnp.full(c.shape[:-1], k, dtype=c.dtype)[..., None],
                k + jnp.cumsum(c, axis=-1) / (n + 1),
            ],
            axis=-1,
        )
        return type(self)(c_new)

    def __call__(self, t: Float[jax.Array, " k"]) -> Float[jax.Array, "*batch k"]:
        r"""Evaluate $p(t)$ with the De Casteljau recurrence.

        Starting from $c_i^{(0)}=c_i$, each reduction level is

        $$
        c_i^{(r+1)}(t)
        =(1-t)c_i^{(r)}(t)+t c_{i+1}^{(r)}(t).
        $$

        After $n$ levels, $c_0^{(n)}(t)=p(t)$. A scalar $t$ is shared across
        leading axes; an array of parameters creates corresponding evaluation
        axes in the result.

        Batch dimensions
        ----------------

        A parameter array does not consume a leading coefficient axis. Its
        shape is appended after all leading axes:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import Bernstein
        >>> Bernstein(jnp.zeros((2, 3)))(jnp.linspace(0.0, 1.0, 4)).shape
        (2, 4)
        >>> Bernstein(jnp.zeros((2, 3, 2)))(jnp.zeros((4, 1))).shape
        (2, 3, 4, 1)
        ```
        """
        c = self.c
        t = jnp.asarray(t, dtype=c.dtype)

        # scalar t -> shared for all batch items
        if t.ndim == 0:
            tt = jnp.broadcast_to(t, c.shape[:-1])[..., None]
            w = c
        else:
            # c: (*batch, order) -> (*batch, 1, order)
            # t: (k,)             -> (1, ..., 1, k, 1)
            # 結果: (*batch, k, order)
            tt = t.reshape((1,) * (c.ndim - 1) + t.shape + (1,))
            w = c.reshape(c.shape[:-1] + (1,) * t.ndim + (c.shape[-1],))

        for _ in range(c.shape[-1] - 1):
            w = (1.0 - tt) * w[..., :-1] + tt * w[..., 1:]

        return w[..., 0]

    def split(self, t: Float[jax.Array, ""] = jnp.array(0.5)) -> tuple[Self, Self]:
        r"""Split $p$ at $t$ into two polynomials reparameterized to $[0,1]$.

        Let $c_i^{(r)}(t)$ be the De Casteljau table described by
        :meth:`__call__`. The left control polygon is

        $$
        (c_0^{(0)},c_0^{(1)}(t),\ldots,c_0^{(n)}(t)),
        $$

        representing $p(ts)$, and the right polygon is the reverse sequence
        of last table entries, representing $p(t+(1-t)s)$ for $s\in[0,1]$.

        Batch dimensions
        ----------------

        The split parameter may provide one value per leading batch item. Both
        returned polynomials retain the original coefficient shape:

        ```python
        >>> import jax.numpy as jnp
        >>> from xbernstein import Bernstein
        >>> left, right = Bernstein(jnp.zeros((2, 3))).split(jnp.array([0.25, 0.75]))
        >>> left.c.shape, right.c.shape
        ((2, 3), (2, 3))
        ```
        """
        c = self.c
        t = jnp.asarray(t, dtype=c.dtype)
        t = jnp.broadcast_to(t, c.shape[:-1])

        tt = t[..., None]
        w = c
        n = c.shape[-1] - 1

        left = [w[..., 0]]
        right = [w[..., -1]]
        for _ in range(n):
            w = (1 - tt) * w[..., :-1] + tt * w[..., 1:]
            left.append(w[..., 0])
            right.append(w[..., -1])

        cl = jnp.stack(left, axis=-1)
        cr = jnp.stack(right[::-1], axis=-1)
        return type(self)(cl), type(self)(cr)


class OptimizeResult(NamedTuple):
    r"""Store an approximation to $\min_{t\in[0,1]}p(t)$ and an associated $t$.

    ``f`` is the current upper-bound value and ``x`` is the parameter at which
    that value is attained. Both arrays have the same leading batch/value
    shape as the input polynomial.

    Batch dimensions
    ----------------

    If the input coefficient array has shape ``(*batch, n + 1)``, both ``f``
    and ``x`` have shape ``(*batch,)``. Each leading index is minimized
    independently.

    ```python
    >>> import jax.numpy as jnp
    >>> result = OptimizeResult(f=jnp.zeros((2, 3)), x=jnp.ones((2, 3)))
    >>> result.f.shape, result.x.shape
    ((2, 3), (2, 3))
    ```
    """

    f: Float[jax.Array, "*batch"]  # 最小値
    x: Float[jax.Array, "*batch"]  # 最小値を与えるx

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the leading shape shared by the minimum value $f$ and point $x$."""
        return self.f.shape

    @property
    def dtype(self) -> str:
        """Return the scalar field used for $f$ and $x$."""
        return str(self.f.dtype)


@jax.custom_jvp
@jaxtyped(typechecker=beartype)
def _minimize(
    coeffs: Float[jax.Array, " n"], max_steps: int = 200, eps: float = 1e-6
) -> tuple[Float[jax.Array, ""], Float[jax.Array, ""]]:
    r"""Approximate $\min_{t\in[0,1]}p(t)$ by Bernstein branch and bound.

    For every active interval $I=[u,v]$, its De Casteljau control polygon
    represents the restricted polynomial. The convex-hull property gives the
    certified lower bound

    $$
    L_I=\min_i c_i^I\leq \min_{t\in I}p(t).
    $$

    Endpoint and midpoint evaluations provide an incumbent upper bound $U$.
    The interval with the smallest $L_I$ is bisected at its midpoint. The
    process stops after ``max_steps`` bisections or when
    $\;U-\min_I L_I\leq\text{``eps``}$.
    """
    a = 0.0
    b = 1.0
    b_init = coeffs
    deg = b_init.shape[0] - 1

    # 必要な最大要素数は max_steps + 1
    max_cap = max_steps + 1

    u_buf = jnp.zeros(max_cap).at[0].set(a)
    v_buf = jnp.zeros(max_cap).at[0].set(b)
    b_buf = jnp.zeros((max_cap, deg + 1)).at[0].set(b_init)
    lb_buf = jnp.full(max_cap, jnp.inf).at[0].set(jnp.min(b_init))

    ub_init = jnp.minimum(b_init[0], b_init[-1])
    best_x_init = jnp.where(b_init[0] < b_init[-1], a, b)

    init_state = (u_buf, v_buf, b_buf, lb_buf, ub_init, best_x_init, 0)

    # 停止条件 (指定ステップ数到達、または許容誤差達成)
    def cond_fn(state):
        r"""Continue while $U-\min_I L_I$ exceeds the requested tolerance."""
        _, _, _, lb_buf, ub, _, step = state
        return (step < max_steps) & ((ub - jnp.min(lb_buf)) > eps)

    # 反復ステップ
    def body_fn(state):
        r"""Bisect the interval attaining $\min_I L_I$ and update the incumbent."""
        u_buf, v_buf, b_buf, lb_buf, ub, best_x, step = state

        min_idx = jnp.argmin(lb_buf)
        u_curr, v_curr, b_curr = u_buf[min_idx], v_buf[min_idx], b_buf[min_idx]

        mid = 0.5 * (u_curr + v_curr)
        left_curve, right_curve = Bernstein(b_curr).split(
            jnp.array(0.5, dtype=b_curr.dtype)
        )
        left_b = left_curve.c
        right_b = right_curve.c

        sample_vals = jnp.array([left_b[0], left_b[-1], right_b[-1]])
        sample_xs = jnp.array([u_curr, mid, v_curr])
        best_sample_idx = jnp.argmin(sample_vals)
        new_ub = jnp.minimum(ub, sample_vals[best_sample_idx])
        new_best_x = jnp.where(
            sample_vals[best_sample_idx] < ub, sample_xs[best_sample_idx], best_x
        )

        # 左ノードを min_idx に上書きし、右ノードを step + 1 に追加
        next_idx = step + 1
        u_buf = u_buf.at[min_idx].set(u_curr).at[next_idx].set(mid)
        v_buf = v_buf.at[min_idx].set(mid).at[next_idx].set(v_curr)
        b_buf = b_buf.at[min_idx].set(left_b).at[next_idx].set(right_b)
        lb_buf = (
            lb_buf.at[min_idx].set(jnp.min(left_b)).at[next_idx].set(jnp.min(right_b))
        )

        return u_buf, v_buf, b_buf, lb_buf, new_ub, new_best_x, step + 1

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return final_state[4], final_state[5]


@_minimize.defjvp
def _minimize_jvp(
    primals: tuple[Float[jax.Array, " n"], int, float],
    tangents: tuple[Float[jax.Array, " n"], int, float],
) -> tuple[
    tuple[Float[jax.Array, ""], Float[jax.Array, ""]],
    tuple[Float[jax.Array, ""], Float[jax.Array, ""]],
]:
    r"""Differentiate the approximate minimum using envelope and implicit rules.

    For coefficient perturbation $\dot p$, the envelope rule treats the
    primal minimizer $x^\ast$ as fixed when differentiating the value:

    $$
    \dot f=\dot p(x^\ast).
    $$

    At an interior stationary minimizer, differentiating
    $p'(x^\ast)=0$ gives

    $$
    \dot x=-\frac{\dot p'(x^\ast)}{p''(x^\ast)}.
    $$

    A boundary minimizer is treated as locally fixed, so its tangent is zero.
    """
    coefficients, max_steps, eps = primals
    coeffs = Bernstein(coefficients)
    t_coeffs, _, _ = tangents
    t_coeffs = Bernstein(t_coeffs)

    primal_out = _minimize(coefficients, max_steps=max_steps, eps=eps)
    x_star = primal_out[1]

    # f(c) = min_t p_c(t) に対して envelope theorem を使い、
    # df ≈ <B(x*), dc> （x* は primal で得た argmin）とする
    tangent_f = t_coeffs(jnp.asarray(x_star, dtype=coeffs.c.dtype))

    # Implicit differentiation of argmin:
    # p'(x*, c) = 0  ->  dx = -(∂p'/∂c · dc) / p''(x*)
    p1 = coeffs.deriv()
    p2 = p1.deriv()
    dp1 = t_coeffs.deriv()

    # x*=0 or x*=1 の場合は境界最小とみなし、argmin の感度は 0 とする。
    # それ以外（内部点）のみ、暗黙微分で dx を計算する。
    tangent_x = jax.lax.cond(
        jnp.equal(x_star, 0.0) | jnp.equal(x_star, 1.0),
        lambda _: jnp.zeros_like(x_star),
        lambda _: -dp1(x_star) / p2(x_star),
        operand=None,
    )

    tangent_out = (tangent_f, tangent_x)
    return primal_out, tangent_out


def minimize(
    bpoly: Float[Bernstein, "*batch"], max_steps: int = 200, eps: float = 1e-6
) -> Float[OptimizeResult, "*batch"]:
    r"""Apply branch-and-bound to each leading-axis polynomial on $[0,1]$.

    The method computes an approximate pair

    $$
    (f,x)\approx\left(\min_{t\in[0,1]}p(t),
    \mathop{\mathrm{argmin}}_{t\in[0,1]}p(t)\right)
    $$

    for every independently batched coefficient vector. It flattens leading
    axes only to vectorize the scalar solver, then restores their original
    arrangement in :class:`OptimizeResult`.

    Batch dimensions
    ----------------

    The final coefficient axis is excluded from the batch shape. The following
    call minimizes two degree-1 polynomials independently:

    ```python
    >>> import jax.numpy as jnp
    >>> from xbernstein import Bernstein
    >>> from xbernstein.bernstein import minimize
    >>> result = minimize(Bernstein(jnp.array([[0.0, 1.0], [1.0, 0.0]])))
    >>> result.f.shape, result.x.shape
    ((2,), (2,))
    ```

    A higher-rank leading shape is likewise preserved:

    ```python
    >>> result = minimize(Bernstein(jnp.zeros((2, 3, 2))))
    >>> result.f.shape
    (2, 3)
    ```

    ``minimize`` accepts one Bernstein object, so it does not broadcast
    separate polynomial inputs. The leading shape is taken entirely from that
    object's coefficient array.
    """
    shape = bpoly.shape
    n = bpoly.order + 1
    coefficients = bpoly.c.reshape([-1, n])

    fs, xs = jax.vmap(_minimize, in_axes=(0, None, None))(coefficients, max_steps, eps)
    results = OptimizeResult(f=fs.reshape(shape), x=xs.reshape(shape))
    return results
