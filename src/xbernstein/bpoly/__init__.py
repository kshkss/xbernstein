import jax
import jax.numpy as jnp
import jax.scipy.special as jss
from jaxtyping import Float, jaxtyped
from beartype import beartype
import equinox as eqx
from typing import Self, NamedTuple


def _elevate(c: jax.Array, target_n: int) -> jax.Array:
    """
    次数昇格（内部用）
    最後の次元を Bernstein 係数次元として、target_n 次まで昇格する。
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
    """Bernstein多項式のクラス"""

    c: Float[jax.Array, "*batch order"]  # 制御点

    @property
    def shape(self) -> tuple[int, ...]:
        return self.c.shape[:-1]

    @property
    def dtype(self) -> str:
        return str(self.c.dtype)

    def __add__(self, other: Self) -> Self:
        """和"""
        c1 = self.c
        c2 = other.c
        target_n = max(c1.shape[-1], c2.shape[-1]) - 1
        _ = jnp.broadcast_shapes(c1.shape[:-1], c2.shape[:-1])
        return type(self)(_elevate(c1, target_n) + _elevate(c2, target_n))

    def __sub__(self, other: Self) -> Self:
        """差"""
        c1 = self.c
        c2 = other.c
        target_n = max(c1.shape[-1], c2.shape[-1]) - 1
        _ = jnp.broadcast_shapes(c1.shape[:-1], c2.shape[:-1])
        return type(self)(_elevate(c1, target_n) - _elevate(c2, target_n))

    def __mul__(self, other: Self) -> Self:
        """積"""
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
        """微分 (m階)"""
        c = self.c
        n = c.shape[-1] - 1

        if m > n:
            c = jnp.zeros(c.shape[:-1] + (1,), dtype=c.dtype)
        else:
            for degree in range(n, n - m, -1):
                c = degree * (c[..., 1:] - c[..., :-1])
        return type(self)(c)

    def int(self, k=0.0) -> Self:
        """積分 (kは積分定数)"""
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

    def __call__(self, t: Float[jax.Array, " k"]) -> Float[jax.Array, "*batch, k"]:
        """ド・カステリョのアルゴリズムによる代入・評価"""
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
        """tでの曲線の分割（De Casteljauのアルゴリズム）"""
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
    f: Float[jax.Array, "*batch"]  # 最小値
    x: Float[jax.Array, "*batch"]  # 最小値を与えるx

    @property
    def shape(self) -> tuple[int, ...]:
        return self.f.shape

    @property
    def dtype(self) -> str:
        return str(self.f.dtype)


@jax.custom_jvp
@jaxtyped(typechecker=beartype)
def _minimize(
    coeffs: Float[jax.Array, " n"], max_steps: int = 200, eps: float = 1e-6
) -> tuple[Float[jax.Array, ""], Float[jax.Array, ""]]:
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
        _, _, _, lb_buf, ub, _, step = state
        return (step < max_steps) & ((ub - jnp.min(lb_buf)) > eps)

    # 反復ステップ
    def body_fn(state):
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
@jaxtyped(typechecker=beartype)
def _minimize_jvp(
    primals: tuple[Float[jax.Array, " n"], int, float],
    tangents: tuple[Float[jax.Array, " n"], int, float],
) -> tuple[
    tuple[Float[jax.Array, ""], Float[jax.Array, ""]],
    tuple[Float[jax.Array, ""], Float[jax.Array, ""]],
]:
    """JVP (Jacobian-vector product) for the _minimize function."""
    coeffs, max_steps, eps = primals
    coeffs = Bernstein(coeffs)
    t_coeffs, _, _ = tangents
    t_coeffs = Bernstein(t_coeffs)

    primal_out = _minimize(coeffs, max_steps=max_steps, eps=eps)
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
    """Bernstein多項式の最小値を求める（JVP対応）"""
    shape = bpoly.shape[:-1]
    n = bpoly.shape[-1]
    coeffs = Bernstein(bpoly.c.reshape([-1, n]))

    fs, xs = jax.vmap(_minimize, in_axes=(0, None, None))(coeffs, max_steps, eps)
    results = OptimizeResult(f=fs.reshape(shape), x=xs.reshape(shape))
    return results
