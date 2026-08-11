r"""Cubic positive-weight rational Hermite interpolation on $[0,1]^d$."""

import itertools
import math

import equinox as eqx
import jax
import jax.numpy as jnp

from .hermite import _multi_indices, _validate_groups
from .rational_bernstein import RationalBernstein
from .rational_tensor_bernstein import (
    RationalBernstein2D,
    RationalBernstein3D,
    RationalBernstein4D,
)


def _univariate_derivative_row(
    derivative_order: int, endpoint: int, dtype
) -> jax.Array:
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


def _derivative_row(alpha, vertex, dtype) -> jax.Array:
    row = jnp.ones(1, dtype=dtype)
    for derivative_order, endpoint in zip(alpha, vertex):
        row = jnp.kron(
            row,
            _univariate_derivative_row(derivative_order, endpoint, dtype),
        )
    return row


def _unpack_derivatives(dimensions: int, values):
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


def _selected_indices(dimensions: int):
    square_free = tuple(itertools.product((0, 1), repeat=dimensions))
    doubled = tuple(
        alpha
        for alpha in itertools.product((0, 2), repeat=dimensions)
        if any(alpha)
    )
    return square_free + doubled


def _multi_binomial(alpha, gamma):
    result = 1
    for upper, lower in zip(alpha, gamma):
        result *= math.comb(upper, lower)
    return result


def _subindices(alpha):
    return itertools.product(*(range(order + 1) for order in alpha))


def _corner_indices(dimensions: int):
    strides = tuple(4 ** (dimensions - axis - 1) for axis in range(dimensions))
    return tuple(
        sum(index * stride for index, stride in zip(corner, strides))
        for corner in itertools.product((0, 3), repeat=dimensions)
    )


def _assemble_system(dimensions: int, derivatives, batch_shape, dtype):
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
        [
            _derivative_row(beta, vertex, dtype)
            for vertex, beta in jet_keys
        ]
    )
    coefficient_from_jet = jnp.linalg.inv(jet_matrix)
    corner_coefficients = _corner_indices(dimensions)
    noncorner_coefficients = tuple(
        index
        for index in range(coefficient_count)
        if index not in set(corner_coefficients)
    )
    noncorner_sum_row = jnp.sum(
        coefficient_from_jet[jnp.asarray(noncorner_coefficients)], axis=0
    )
    normalization_row = jnp.concatenate(
        [
            jnp.zeros(coefficient_count, dtype=dtype),
            noncorner_sum_row[jnp.asarray(derivative_jet_indices)],
        ]
    )
    normalization_rhs = 1.0 - jnp.sum(
        noncorner_sum_row[jnp.asarray(value_jet_indices)]
    )
    unknown_count = coefficient_count + len(derivative_jet_indices)
    rows = []
    right_hand_sides = []
    derivative_rows = {
        (alpha, vertex): (
            _derivative_row(alpha, vertex, dtype) @ coefficient_from_jet
        )
        for alpha in itertools.product(range(3), repeat=dimensions)
        for vertex in vertices
    }

    for vertex in vertices:
        vertex_index = (...,) + vertex
        for alpha in _selected_indices(dimensions):
            denominator_row = jnp.zeros(coefficient_count, dtype=dtype)
            for gamma in _subindices(alpha):
                remaining = tuple(
                    upper - lower for upper, lower in zip(alpha, gamma)
                )
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
        normalization_row,
        normalization_rhs,
    )


def _matrix_rank(matrix: jax.Array) -> jax.Array:
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    tolerance = (
        jnp.finfo(matrix.dtype).eps
        * jnp.maximum(singular_values[0], 1.0)
    )
    return jnp.sum(singular_values > tolerance)


def _solve_one(
    matrix: jax.Array,
    right_hand_side: jax.Array,
    normalization_row: jax.Array,
    normalization_rhs: jax.Array,
):
    unknown_count = matrix.shape[-1]
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
    rank = _matrix_rank(scaled_matrix)
    normalization_scale = jnp.maximum(
        jnp.max(jnp.abs(normalization_row)),
        jnp.maximum(jnp.abs(normalization_rhs), 1.0),
    )
    scaled_normalization = (
        normalization_row / normalization_scale / column_scale
    )
    augmented_matrix = jnp.concatenate(
        [scaled_matrix, scaled_normalization[None, :]], axis=0
    )
    augmented_rhs = jnp.concatenate(
        [normalized_rhs, jnp.atleast_1d(normalization_rhs / normalization_scale)]
    )
    augmented_rank = _matrix_rank(augmented_matrix)

    def solve_full(_):
        return jnp.linalg.solve(scaled_matrix, normalized_rhs)

    def solve_augmented(_):
        return jnp.linalg.lstsq(
            augmented_matrix, augmented_rhs, rcond=None
        )[0]

    scaled_solution = jax.lax.cond(
        rank == unknown_count,
        solve_full,
        solve_augmented,
        operand=None,
    )
    solution = scaled_solution / column_scale
    residual = normalized_matrix @ solution - normalized_rhs
    residual_tolerance = (
        10.0 * jnp.sqrt(jnp.finfo(matrix.dtype).eps)
    )
    invalid = (
        ((rank < unknown_count) & (augmented_rank < unknown_count))
        | (jnp.max(jnp.abs(residual)) > residual_tolerance)
        | ~jnp.all(jnp.isfinite(solution))
    )
    return solution, invalid


def _interpolate(dimensions: int, groups):
    values, batch_shape = _validate_groups(dimensions, 5, groups)
    derivatives = _unpack_derivatives(dimensions, values)
    dtype = values[0].dtype
    (
        matrix,
        right_hand_side,
        coefficient_from_jet,
        value_jet_indices,
        derivative_jet_indices,
        normalization_row,
        normalization_rhs,
    ) = _assemble_system(dimensions, derivatives, batch_shape, dtype)
    coefficient_count = 4**dimensions
    flat_matrix = matrix.reshape((-1,) + matrix.shape[-2:])
    flat_rhs = right_hand_side.reshape((-1, right_hand_side.shape[-1]))
    solutions, invalid = jax.vmap(_solve_one, in_axes=(0, 0, None, None))(
        flat_matrix,
        flat_rhs,
        normalization_row,
        normalization_rhs,
    )
    solutions = eqx.error_if(
        solutions,
        jnp.any(invalid),
        "rational Hermite interpolation system is singular or inconsistent",
    )
    solutions = solutions.reshape(batch_shape + (solutions.shape[-1],))

    numerator_jets = solutions[..., :coefficient_count]
    denominator_jets = jnp.ones(batch_shape + (coefficient_count,), dtype=dtype)
    denominator_jets = denominator_jets.at[..., derivative_jet_indices].set(
        solutions[..., coefficient_count:]
    )
    numerator = jnp.einsum(
        "ij,...j->...i", coefficient_from_jet, numerator_jets
    )
    denominator = jnp.einsum(
        "ij,...j->...i", coefficient_from_jet, denominator_jets
    )
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


def rational_hermite_interpolate_1d(f, d1, d2):
    r"""Interpolate $f(x)$ with a degree-3 :class:`RationalBernstein`.

    **Inputs:** ``f[...,vx]=f(vx)``, ``d1[...,vx]=f_x(vx)``, and
    ``d2[...,vx]=f_xx(vx)``; all have shape ``(*batch,2)``. Common leading
    batch axes are preserved and each batch item is solved independently.

    Writing the result as $R=N/D$, the equations follow
    $\partial^\alpha N=\sum_{\gamma\leq\alpha}\binom{\alpha}{\gamma}
    (\partial^\gamma f)(\partial^{\alpha-\gamma}D)$. Thus $R$, $R_x$, and
    $R_{xx}$ reproduce the supplied endpoint data. Endpoint denominator
    weights are fixed to one.

    **Returns:** ``rpoly.order == 3``. A singular or inconsistent system, or
    any non-positive reconstructed denominator weight, raises an error.
    """
    return _interpolate(1, (f, d1, d2))


def rational_hermite_interpolate_2d(f, d1, d2, d3, d4):
    r"""Interpolate $f(x,y)$ with a degree-$(3,3)$ rational Bernstein.

    **Inputs:** vertex axes are ``vx,vy``:
    ``f[...,vx,vy]=f(vx,vy)``.
    ``d1[...,vx,vy,0/1]=(f_x,f_y)``,
    ``d2[...,vx,vy,0/1/2]=(f_xx,f_xy,f_yy)``,
    ``d3[...,vx,vy,0/1]=(f_xxy,f_xyy)``, and
    ``d4[...,vx,vy]=f_xxyy``. Common leading batch axes are preserved and
    each batch item is solved independently.

    Writing the result as $R=N/D$, it reproduces every vertex derivative
    $\partial^\alpha f$ for
    $\alpha\in\{0,1\}^2\cup(\{0,2\}^2\setminus\{(0,0)\})$: the value,
    gradient, Hessian, and $f_{xxyy}$. Every input is required; the Leibniz
    equation for $R_{xxyy}$ also contains $f_{xxy}$ and $f_{xyy}$. Corner
    denominator weights are fixed to one.

    **Returns:** ``rpoly.order == [3,3]``. A singular or inconsistent system,
    or any non-positive reconstructed weight, raises an error.
    """
    return _interpolate(2, (f, d1, d2, d3, d4))


def rational_hermite_interpolate_3d(f, d1, d2, d3, d4, d5, d6):
    r"""Interpolate $f(x,y,z)$ with a degree-$(3,3,3)$ rational Bernstein.

    **Inputs:** vertex axes are ``vx,vy,vz``:
    ``f[...,vx,vy,vz]=f(vx,vy,vz)``.
    ``d1[...,vx,vy,vz,0/1/2]=(f_x,f_y,f_z)``,
    ``d2[...,vx,vy,vz,0/1/2/3/4/...]=(f_xx,f_xy,f_xz,f_yy,f_yz,...)``,
    ``d3[...,vx,vy,vz,0/1/2/3/4/...]=(f_xxy,f_xxz,f_xyy,f_xyz,f_xzz,...)``,
    ``d4[...,vx,vy,vz,0/1/2/3/4/...]=(f_xxyy,f_xxyz,f_xxzz,f_xyyz,f_xyzz,...)``,
    ``d5[...,vx,vy,vz,0/1/2]=(f_xxyyz,f_xxyzz,f_xyyzz)``, and
    ``d6[...,vx,vy,vz]=f_xxyyzz``. Derivative-kind counts are
    ``3,6,7,6,3,1``. Common leading batch axes are preserved and solved
    independently.

    The result reproduces every vertex derivative $\partial^\alpha f$ for
    $\alpha\in\{0,1\}^3\cup(\{0,2\}^3\setminus\{(0,0,0)\})$. Every other
    supplied derivative is still used in the Leibniz equations for these
    conditions. Corner denominator weights are fixed to one.

    **Returns:** ``rpoly.order == [3,3,3]``. A singular or inconsistent
    system, or any non-positive reconstructed weight, raises an error.
    """
    return _interpolate(3, (f, d1, d2, d3, d4, d5, d6))


def rational_hermite_interpolate_4d(f, d1, d2, d3, d4, d5, d6, d7, d8):
    r"""Interpolate $f(x,y,z,w)$ with a degree-$(3,3,3,3)$ rational Bernstein.

    **Inputs:** vertex axes are ``vx,vy,vz,vw``:
    ``f[...,vx,vy,vz,vw]=f(vx,vy,vz,vw)``.
    ``d1[...,vx,vy,vz,vw,0/1/2/3]=(f_x,f_y,f_z,f_w)``,
    ``d2[...,vx,vy,vz,vw,0/1/2/3/4/...]=(f_xx,f_xy,f_xz,f_xw,f_yy,...)``,
    ``d3[...,vx,vy,vz,vw,0/1/2/3/4/...]=(f_xxy,f_xxz,f_xxw,f_xyy,f_xyz,...)``,
    ``d4[...,vx,vy,vz,vw,0/1/2/3/4/...]=(f_xxyy,f_xxyz,f_xxyw,f_xxzz,f_xxzw,...)``,
    ``d5[...,vx,vy,vz,vw,0/1/2/3/4/...]=(f_xxyyz,f_xxyyw,f_xxyzz,f_xxyzw,f_xxyww,...)``,
    ``d6[...,vx,vy,vz,vw,0/1/2/3/4/...]=(f_xxyyzz,f_xxyyzw,f_xxyyww,f_xxyzzw,f_xxyzww,...)``,
    ``d7[...,vx,vy,vz,vw,0/1/2/3]=(f_xxyyzzw,f_xxyyzww,f_xxyzzww,f_xyyzzww)``,
    and ``d8[...,vx,vy,vz,vw]=f_xxyyzzww``. Derivative-kind counts are
    ``4,10,16,19,16,10,4,1``. Common leading batch axes are preserved and
    solved independently.

    The result reproduces every vertex derivative $\partial^\alpha f$ for
    $\alpha\in\{0,1\}^4\cup(\{0,2\}^4\setminus\{(0,0,0,0)\})$. Every other
    supplied derivative is still used in the Leibniz equations for these
    conditions. Corner denominator weights are fixed to one.

    **Returns:** ``rpoly.order == [3,3,3,3]``. A singular or inconsistent
    system, or any non-positive reconstructed weight, raises an error.
    """
    return _interpolate(4, (f, d1, d2, d3, d4, d5, d6, d7, d8))
