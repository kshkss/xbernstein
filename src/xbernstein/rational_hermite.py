r"""Cubic positive-weight rational Hermite interpolation on $[0,1]^d$."""

import itertools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float, Int, Shaped

from .hermite import _multi_indices, _validate_groups
from .rational_bernstein import RationalBernstein
from .rational_tensor_bernstein import (
    RationalBernstein2D,
    RationalBernstein3D,
    RationalBernstein4D,
)


def _univariate_derivative_row(
    derivative_order: int, endpoint: int, dtype
) -> Float[jax.Array, "4"]:
    degree = 3
    row = jnp.zeros(degree + 1, dtype=dtype)
    scale = math.factorial(degree) / math.factorial(degree - derivative_order)
    start = 0 if endpoint == 0 else degree - derivative_order
    for index in range(derivative_order + 1):
        coefficient = (
            scale
            * (-1) ** (derivative_order - index)
            * math.comb(derivative_order, index)
        )
        row = row.at[start + index].set(coefficient)
    return row


def _derivative_row(
    alpha: tuple[int, ...], vertex: tuple[int, ...], dtype
) -> Float[jax.Array, "n"]:
    row = jnp.ones(1, dtype=dtype)
    for derivative_order, endpoint in zip(alpha, vertex):
        row = jnp.kron(
            row,
            _univariate_derivative_row(derivative_order, endpoint, dtype),
        )
    return row


def _unpack_derivatives(
    dimensions: int, values: tuple[Float[jax.Array, "..."] , ...]
) -> dict[tuple[int, ...], Float[jax.Array, "..."]]:
    derivatives = {}
    for total, group in enumerate(values):
        indices = _multi_indices(dimensions, 3, total)
        if total == 0:
            derivatives[indices[0]] = group
        elif len(indices) == 1:
            derivatives[indices[0]] = group
        else:
            for derivative_index, alpha in enumerate(indices):
                derivatives[alpha] = group[..., derivative_index]
    return derivatives


def _selected_indices(dimensions: int) -> tuple[tuple[int, ...], ...]:
    square_free = tuple(itertools.product((0, 1), repeat=dimensions))
    doubled = tuple(
        alpha for alpha in itertools.product((0, 2), repeat=dimensions) if any(alpha)
    )
    return square_free + doubled


def _multi_binomial(alpha: tuple[int, ...], gamma: tuple[int, ...]) -> int:
    result = 1
    for upper, lower in zip(alpha, gamma):
        result *= math.comb(upper, lower)
    return result


def _subindices(alpha: tuple[int, ...]):
    return itertools.product(*(range(order + 1) for order in alpha))


def _assemble_system(
    dimensions: int,
    derivatives: dict[tuple[int, ...], Float[jax.Array, "..."]],
    batch_shape: tuple[int, ...],
    dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple[int, ...], tuple[int, ...]]:
    coefficient_count = 4**dimensions
    vertices = tuple(itertools.product((0, 1), repeat=dimensions))
    square_free = tuple(itertools.product((0, 1), repeat=dimensions))
    jet_keys = tuple(itertools.product(vertices, square_free))
    value_jet_indices = tuple(
        index for index, (_, beta) in enumerate(jet_keys) if not any(beta)
    )
    derivative_jet_indices = tuple(
        index for index in range(coefficient_count) if index not in value_jet_indices
    )
    jet_matrix = jnp.stack(
        [_derivative_row(beta, vertex, dtype) for vertex, beta in jet_keys]
    )
    coefficient_from_jet = jnp.linalg.inv(jet_matrix)
    unknown_count = coefficient_count + len(derivative_jet_indices)
    rows = []
    right_hand_sides = []
    derivative_rows = {
        (alpha, vertex): (_derivative_row(alpha, vertex, dtype) @ coefficient_from_jet)
        for alpha in itertools.product(range(3), repeat=dimensions)
        for vertex in vertices
    }

    for vertex in vertices:
        vertex_index = (...,) + vertex
        for alpha in _selected_indices(dimensions):
            denominator_row = jnp.zeros(coefficient_count, dtype=dtype)
            for gamma in _subindices(alpha):
                remaining = tuple(upper - lower for upper, lower in zip(alpha, gamma))
                denominator_row = denominator_row - (
                    _multi_binomial(alpha, gamma)
                    * derivatives[gamma][vertex_index][..., None]
                    * derivative_rows[(remaining, vertex)]
                )

            numerator_row = derivative_rows[(alpha, vertex)]
            numerator_row = jnp.broadcast_to(
                numerator_row, batch_shape + (coefficient_count,)
            )
            row = jnp.concatenate(
                [
                    numerator_row,
                    denominator_row[..., derivative_jet_indices],
                ],
                axis=-1,
            )
            rows.append(row)
            right_hand_sides.append(
                -jnp.sum(denominator_row[..., value_jet_indices], axis=-1)
            )

    matrix = jnp.stack(rows, axis=-2)
    right_hand_side = jnp.stack(right_hand_sides, axis=-1)
    if matrix.shape[-2:] != (unknown_count, unknown_count):
        raise RuntimeError("rational Hermite system must be square")
    return (
        matrix,
        right_hand_side,
        coefficient_from_jet,
        value_jet_indices,
        derivative_jet_indices,
    )


def _solve_subset_one(
    matrix: Float[jax.Array, "rows columns"],
    right_hand_side: Float[jax.Array, "rows"],
) -> tuple[Float[jax.Array, "columns"], Bool[jax.Array, ""]]:
    """Return a scaled minimum-norm solution for one axis subset."""
    row_scale = jnp.maximum(
        jnp.max(jnp.abs(matrix), axis=-1),
        jnp.maximum(jnp.abs(right_hand_side), 1.0),
    )
    normalized_matrix = matrix / row_scale[:, None]
    normalized_rhs = right_hand_side / row_scale
    column_scale = jnp.maximum(
        jnp.max(jnp.abs(normalized_matrix), axis=0),
        jnp.finfo(matrix.dtype).eps,
    )
    scaled_matrix = normalized_matrix / column_scale[None, :]
    singular_values = jnp.linalg.svd(scaled_matrix, compute_uv=False)
    rank_tolerance = jnp.finfo(matrix.dtype).eps * jnp.maximum(
        singular_values[0], 1.0
    )
    rank = jnp.sum(singular_values > rank_tolerance)

    def solve_regular(_):
        return jnp.linalg.solve(scaled_matrix, normalized_rhs)

    def solve_rank_deficient(_):
        return jnp.linalg.lstsq(scaled_matrix, normalized_rhs, rcond=None)[0]

    scaled_solution = jax.lax.cond(
        rank == matrix.shape[-1],
        solve_regular,
        solve_rank_deficient,
        operand=None,
    )
    solution = scaled_solution / column_scale
    residual = normalized_matrix @ solution - normalized_rhs
    residual_tolerance = 10.0 * jnp.sqrt(jnp.finfo(matrix.dtype).eps)
    invalid = (
        (jnp.max(jnp.abs(residual)) > residual_tolerance)
        | ~jnp.all(jnp.isfinite(solution))
    )
    return solution, invalid


def _axis_subsets(dimensions: int):
    """Yield non-empty coordinate subsets by size and lexicographic order."""
    for size in range(1, dimensions + 1):
        yield from itertools.combinations(range(dimensions), size)


def _subset_indices(
    dimensions: int, axes: tuple[int, ...], derivative_jet_indices: tuple[int, ...]
) -> tuple[Int[jax.Array, "rows"], Int[jax.Array, "columns"]]:
    """Return rows and unknown columns for exactly one coordinate subset."""
    vertices = tuple(itertools.product((0, 1), repeat=dimensions))
    square_free = tuple(itertools.product((0, 1), repeat=dimensions))
    jet_keys = tuple(itertools.product(vertices, square_free))
    coefficient_count = len(jet_keys)
    derivative_columns = {
        jet_index: coefficient_count + derivative_index
        for derivative_index, jet_index in enumerate(derivative_jet_indices)
    }
    selected = _selected_indices(dimensions)
    support = tuple(int(axis in axes) for axis in range(dimensions))
    doubled_support = tuple(2 * order for order in support)
    rows = []
    columns = []
    for vertex_index, _ in enumerate(vertices):
        row_offset = vertex_index * len(selected)
        for alpha_index, alpha in enumerate(selected):
            if alpha == support or alpha == doubled_support:
                rows.append(row_offset + alpha_index)
        for beta_index, beta in enumerate(square_free):
            if beta == support:
                jet_index = vertex_index * len(square_free) + beta_index
                columns.extend((jet_index, derivative_columns[jet_index]))
    return jnp.asarray(rows, dtype=jnp.int32), jnp.asarray(columns, dtype=jnp.int32)


def _interpolate(
    dimensions: int, groups: tuple[Float[jax.Array, "..."], ...]
) -> Shaped[RationalBernstein, "*batch"] | Shaped[RationalBernstein2D, "*batch"] | Shaped[RationalBernstein3D, "*batch"] | Shaped[RationalBernstein4D, "*batch"]:
    values, batch_shape = _validate_groups(dimensions, 5, groups)
    derivatives = _unpack_derivatives(dimensions, values)
    dtype = values[0].dtype
    (
        matrix,
        right_hand_side,
        coefficient_from_jet,
        value_jet_indices,
        derivative_jet_indices,
    ) = _assemble_system(dimensions, derivatives, batch_shape, dtype)
    coefficient_count = 4**dimensions
    solutions = jnp.zeros(batch_shape + (matrix.shape[-1],), dtype=dtype)
    vertex_values = values[0].reshape(batch_shape + (2**dimensions,))
    solutions = solutions.at[..., jnp.asarray(value_jet_indices)].set(vertex_values)
    invalid = jnp.array(False)
    for axes in _axis_subsets(dimensions):
        rows, columns = _subset_indices(dimensions, axes, derivative_jet_indices)
        stage_matrix = jnp.take(jnp.take(matrix, rows, axis=-2), columns, axis=-1)
        known = jnp.take(matrix, rows, axis=-2) @ solutions[..., None]
        stage_rhs = right_hand_side[..., rows] - known[..., 0]
        flat_matrix = stage_matrix.reshape((-1,) + stage_matrix.shape[-2:])
        flat_rhs = stage_rhs.reshape((-1, stage_rhs.shape[-1]))
        stage_solutions, stage_invalid = jax.vmap(_solve_subset_one)(
            flat_matrix, flat_rhs
        )
        stage_solutions = stage_solutions.reshape(
            batch_shape + (stage_solutions.shape[-1],)
        )
        solutions = solutions.at[..., columns].set(stage_solutions)
        invalid = invalid | jnp.any(stage_invalid)
    full_residual = matrix @ solutions[..., None] - right_hand_side[..., None]
    full_scale = jnp.maximum(
        jnp.max(jnp.abs(matrix), axis=-1),
        jnp.maximum(jnp.abs(right_hand_side), 1.0),
    )
    invalid = invalid | jnp.any(
        jnp.abs(full_residual[..., 0]) / full_scale
        > 10.0 * jnp.sqrt(jnp.finfo(dtype).eps)
    )
    solutions = eqx.error_if(
        solutions,
        invalid,
        "rational Hermite interpolation stage is inconsistent",
    )

    numerator_jets = solutions[..., :coefficient_count]
    denominator_jets = jnp.ones(batch_shape + (coefficient_count,), dtype=dtype)
    denominator_jets = denominator_jets.at[..., derivative_jet_indices].set(
        solutions[..., coefficient_count:]
    )
    numerator = jnp.einsum("ij,...j->...i", coefficient_from_jet, numerator_jets)
    denominator = jnp.einsum("ij,...j->...i", coefficient_from_jet, denominator_jets)
    denominator = eqx.error_if(
        denominator,
        jnp.any(denominator <= 0.0),
        "rational Hermite interpolation produced non-positive weights",
    )
    coefficient_shape = (4,) * dimensions
    numerator = numerator.reshape(batch_shape + coefficient_shape)
    denominator = denominator.reshape(batch_shape + coefficient_shape)
    values = numerator / denominator
    if dimensions == 1:
        return RationalBernstein(values, denominator)
    if dimensions == 2:
        return RationalBernstein2D(values, denominator)
    if dimensions == 3:
        return RationalBernstein3D(values, denominator)
    if dimensions == 4:
        return RationalBernstein4D(values, denominator)
    raise ValueError(f"unsupported dimension: {dimensions}")


def rational_hermite_interpolate_1d(
    f: Float[jax.Array, "*batch 2"],
    d1: Float[jax.Array, "*batch 2"],
    d2: Float[jax.Array, "*batch 2"],
) -> Shaped[RationalBernstein, "*batch"]:
    r"""Interpolate endpoint values through second derivatives by a cubic rational.

    The result has the form

    $$
    R(x)=\frac{N(x)}{D(x)},\qquad
    N(x)=\sum_{i=0}^{3}n_iB_i^3(x),\qquad
    D(x)=\sum_{i=0}^{3}w_iB_i^3(x),
    $$

    and satisfies

    $$
    R^{(k)}(v)=f^{(k)}(v),
    \qquad v\in\{0,1\},\quad k\in\{0,1,2\}.
    $$

    Thus every supplied value, first derivative, and second derivative is an
    exact interpolation condition and is reproduced exactly. The coefficients
    are found by differentiating $N=fD$ at each endpoint:

    $$
    N^{(k)}(v)=\sum_{j=0}^{k}\binom{k}{j}
    f^{(j)}(v)D^{(k-j)}(v).
    $$

    The endpoint denominator controls are normalized to $w_0=w_3=1$, and all
    reconstructed $w_i$ must be positive; consequently $D(x)>0$ on
    $[0,1]$.

    **Inputs:** $f[...,v_x]=f(v_x)$,
    $d_1[...,v_x]=f_x(v_x)$, and $d_2[...,v_x]=f_{xx}(v_x)$; all have shape
    ``(*batch,2)``. Each leading batch item is solved independently.
    **Returns:** $\operatorname{order}(R)=3$. The rank-deficient stage, if any, uses
    a scaled minimum-norm solution. An inconsistent stage or any non-positive
    reconstructed denominator weight raises an error.
    """
    return _interpolate(1, (f, d1, d2))


def rational_hermite_interpolate_2d(
    f: Float[jax.Array, "*batch 2 2"],
    d1: Float[jax.Array, "*batch 2 2 2"],
    d2: Float[jax.Array, "*batch 2 2 3"],
    d3: Float[jax.Array, "*batch 2 2 2"],
    d4: Float[jax.Array, "*batch 2 2"],
) -> Shaped[RationalBernstein2D, "*batch"]:
    r"""Interpolate selected vertex derivatives by a bicubic rational function.

    The result is

    $$
    R(\mathbf{x})=\frac{N(\mathbf{x})}{D(\mathbf{x})},
    \qquad N,D\in\mathcal Q_3,
    $$

    where $\mathcal Q_3$ is the tensor-product space of degree at most three
    in each coordinate. More explicitly,

    $$
    N(x,y)=\sum_{i,j=0}^{3}n_{ij}B_i^3(x)B_j^3(y),
    \qquad
    D(x,y)=\sum_{i,j=0}^{3}w_{ij}B_i^3(x)B_j^3(y).
    $$

    At every vertex, $R$ satisfies

    $$
    \partial^\alpha R(\mathbf v)=\partial^\alpha f(\mathbf v),
    \qquad
    \mathbf v\in\{0,1\}^2,\quad
    \alpha\in\mathcal S_2,
    $$

    with

    $$
    \mathcal S_2=
    \{0,1\}^2\cup\left(\{0,2\}^2\setminus\{(0,0)\}\right).
    $$

    These conditions reproduce the value, gradient, full Hessian, and
    $f_{xxyy}$. The supplied $f_{xxy}$ and $f_{xyy}$ are not themselves
    reproduction conditions, but they are still required. The coefficients
    follow from the vertex equations obtained by differentiating $N=fD$:

    $$
    \partial^\alpha N(\mathbf v)=
    \sum_{\gamma\leq\alpha}\binom{\alpha}{\gamma}
    \partial^\gamma f(\mathbf v)
    \partial^{\alpha-\gamma}D(\mathbf v).
    $$

    In particular, the equation for $\alpha=(2,2)$ contains both auxiliary
    third derivatives. Corner denominator controls are fixed to one and all
    denominator controls must be positive, so $D>0$ on $[0,1]^2$.

    **Inputs:** vertex axes are $v_x,v_y$:
    $f[...,v_x,v_y]=f(v_x,v_y)$.
    $d_1[...,v_x,v_y,0/1]=(f_x,f_y)$,
    $d_2[...,v_x,v_y,0/1/2]=(f_{xx},f_{xy},f_{yy})$,
    $d_3[...,v_x,v_y,0/1]=(f_{xxy},f_{xyy})$, and
    $d_4[...,v_x,v_y]=f_{xxyy}$. Each leading batch item is solved
    independently.
    The solver fixes the ``x`` and ``y`` terms independently, then the
    ``xy`` crossing terms. Rank-deficient subset systems use scaled
    minimum-norm solutions.
    **Returns:** $\operatorname{order}(R)=(3,3)$. An inconsistent stage or any
    non-positive reconstructed weight raises an error.
    """
    return _interpolate(2, (f, d1, d2, d3, d4))


def rational_hermite_interpolate_3d(
    f: Float[jax.Array, "*batch 2 2 2"],
    d1: Float[jax.Array, "*batch 2 2 2 3"],
    d2: Float[jax.Array, "*batch 2 2 2 6"],
    d3: Float[jax.Array, "*batch 2 2 2 7"],
    d4: Float[jax.Array, "*batch 2 2 2 6"],
    d5: Float[jax.Array, "*batch 2 2 2 3"],
    d6: Float[jax.Array, "*batch 2 2 2"],
) -> Shaped[RationalBernstein3D, "*batch"]:
    r"""Interpolate selected vertex derivatives by a tricubic rational function.

    The result is a ratio of tensor-product cubics,

    $$
    R(\mathbf{x})=\frac{N(\mathbf{x})}{D(\mathbf{x})},
    \qquad
    N,D\in\mathcal Q_3,\qquad
    P(\mathbf{x})
    =\sum_{\mathbf i\in\{0,\ldots,3\}^3}
    p_{\mathbf i}\prod_{a=1}^{3}B_{i_a}^3(x_a),
    \quad P\in\{N,D\}.
    $$

    Its exact vertex interpolation conditions are

    $$
    \partial^\alpha R(\mathbf v)=\partial^\alpha f(\mathbf v),
    \qquad
    \mathbf v\in\{0,1\}^3,\quad
    \alpha\in\mathcal S_3,
    $$

    where

    $$
    \mathcal S_3=
    \{0,1\}^3
    \cup\left(\{0,2\}^3\setminus\{(0,0,0)\}\right).
    $$

    Hence all square-free derivatives and all nonzero doubled-axis
    derivatives are reproduced, including $f_{xyz}$, the pure second
    derivatives, the doubled-axis pair derivatives, and $f_{xxyyzz}$.
    Derivatives supplied for multi-indices outside $\mathcal S_3$ are
    auxiliary data, not additional reproduction conditions. They enter the
    linear equations obtained from

    $$
    \partial^\alpha N(\mathbf v)=
    \sum_{\gamma\leq\alpha}\binom{\alpha}{\gamma}
    \partial^\gamma f(\mathbf v)
    \partial^{\alpha-\gamma}D(\mathbf v).
    $$

    This is why every input group below is required. Corner denominator
    controls are normalized to one; all reconstructed denominator controls
    must be positive, ensuring $D>0$ on $[0,1]^3$.

    **Inputs:** vertex axes are $v_x,v_y,v_z$:
    $f[...,v_x,v_y,v_z]=f(v_x,v_y,v_z)$.
    $d_1[...,v_x,v_y,v_z,0/1/2]=(f_x,f_y,f_z)$,
    $d_2[...,v_x,v_y,v_z,0/1/2/3/4]=(f_{xx},f_{xy},f_{xz},f_{yy},f_{yz},...)$,
    $d_3[...,v_x,v_y,v_z,0/1/2/3/4]=(f_{xxy},f_{xxz},f_{xyy},f_{xyz},f_{xzz},...)$,
    $d_4[...,v_x,v_y,v_z,0/1/2/3/4]=(f_{xxyy},f_{xxyz},f_{xxzz},f_{xyyz},f_{xyzz},...)$,
    $d_5[...,v_x,v_y,v_z,0/1/2]=(f_{xxyyz},f_{xxyzz},f_{xyyzz})$, and
    $d_6[...,v_x,v_y,v_z]=f_{xxyyzz}$. Derivative-kind counts are
    ``3,6,7,6,3,1``. Each leading batch item is solved independently.
    Terms are solved independently for each axis subset, ordered by subset
    size; rank-deficient subset systems use scaled minimum-norm solutions.
    **Returns:** $\operatorname{order}(R)=(3,3,3)$. An inconsistent stage or any
    non-positive reconstructed weight raises an error.
    """
    return _interpolate(3, (f, d1, d2, d3, d4, d5, d6))


def rational_hermite_interpolate_4d(
    f: Float[jax.Array, "*batch 2 2 2 2"],
    d1: Float[jax.Array, "*batch 2 2 2 2 4"],
    d2: Float[jax.Array, "*batch 2 2 2 2 10"],
    d3: Float[jax.Array, "*batch 2 2 2 2 16"],
    d4: Float[jax.Array, "*batch 2 2 2 2 19"],
    d5: Float[jax.Array, "*batch 2 2 2 2 16"],
    d6: Float[jax.Array, "*batch 2 2 2 2 10"],
    d7: Float[jax.Array, "*batch 2 2 2 2 4"],
    d8: Float[jax.Array, "*batch 2 2 2 2"],
) -> Shaped[RationalBernstein4D, "*batch"]:
    r"""Interpolate selected vertex derivatives by a 4D tensor-cubic rational.

    The result is

    $$
    R(\mathbf{x})=\frac{N(\mathbf{x})}{D(\mathbf{x})},
    \qquad
    N,D\in\mathcal Q_3,\qquad
    P(\mathbf{x})
    =\sum_{\mathbf i\in\{0,\ldots,3\}^4}
    p_{\mathbf i}\prod_{a=1}^{4}B_{i_a}^3(x_a),
    \quad P\in\{N,D\}.
    $$

    Thus numerator and denominator both have degree at most three in each
    coordinate. The exact interpolation conditions are

    $$
    \partial^\alpha R(\mathbf v)=\partial^\alpha f(\mathbf v),
    \qquad
    \mathbf v\in\{0,1\}^4,\quad
    \alpha\in\mathcal S_4,
    $$

    with

    $$
    \mathcal S_4=
    \{0,1\}^4
    \cup\left(\{0,2\}^4\setminus\{(0,0,0,0)\}\right).
    $$

    This reproduces all square-free derivatives and every nonzero derivative
    in which each coordinate is used zero or two times, through
    $f_{xxyyzzww}$. Other supplied derivatives are auxiliary: they are not
    reproduced, but are needed in the product-rule equations

    $$
    \partial^\alpha N(\mathbf v)=
    \sum_{\gamma\leq\alpha}\binom{\alpha}{\gamma}
    \partial^\gamma f(\mathbf v)
    \partial^{\alpha-\gamma}D(\mathbf v)
    $$

    used to solve the denominator. Therefore none of the input groups below
    can be omitted. Corner denominator controls are normalized to one and all
    reconstructed controls must be positive, so $D>0$ on $[0,1]^4$.

    **Inputs:** vertex axes are $v_x,v_y,v_z,v_w$:
    $f[...,v_x,v_y,v_z,v_w]=f(v_x,v_y,v_z,v_w)$.
    $d_1[...,v_x,v_y,v_z,v_w,0/1/2/3]=(f_x,f_y,f_z,f_w)$,
    ``d2[...,v_x,v_y,v_z,v_w,0/1/2/3/4/...]=(f_xx,f_xy,f_xz,f_xw,f_yy,...)``,
    ``d3[...,v_x,v_y,v_z,v_w,0/1/2/3/4/...]=(f_xxy,f_xxz,f_xxw,f_xyy,f_xyz,...)``,
    ``d4[...,v_x,v_y,v_z,v_w,0/1/2/3/4/...]=(f_xxyy,f_xxyz,f_xxyw,f_xxzz,f_xxzw,...)``,
    ``d5[...,v_x,v_y,v_z,v_w,0/1/2/3/4/...]=(f_xxyyz,f_xxyyw,f_xxyzz,f_xxyzw,f_xxyww,...)``,
    ``d6[...,v_x,v_y,v_z,v_w,0/1/2/3/4/...]=(f_xxyyzz,f_xxyyzw,f_xxyyww,f_xxyzzw,f_xxyzww,...)``,
    $d_7[...,v_x,v_y,v_z,v_w,0/1/2/3]=(f_{xxyyzzw},f_{xxyyzww},f_{xxyzzww},f_{xyyzzww})$,
    and $d_8[...,v_x,v_y,v_z,v_w]=f_{xxyyzzww}$. Derivative-kind counts are
    ``4,10,16,19,16,10,4,1``. Each leading batch item is solved
    independently.
    Terms are solved independently for each axis subset, ordered by subset
    size; a rank-deficient subset system uses a scaled minimum-norm solution.
    **Returns:** $\operatorname{order}(R)=(3,3,3,3)$. An inconsistent stage or any
    non-positive reconstructed weight raises an error.
    """
    return _interpolate(4, (f, d1, d2, d3, d4, d5, d6, d7, d8))
