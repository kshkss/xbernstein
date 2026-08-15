# Bernstein basis

The one-dimensional Bernstein basis of degree $n$ is

$$
B_i^n(t)=\binom ni t^i(1-t)^{n-i},
\qquad 0\le t\le1.
$$

The polynomial represented by coefficients $c_0,\ldots,c_n$ is

$$
p(t)=\sum_{i=0}^{n}c_iB_i^n(t).
$$

The basis is nonnegative and forms a partition of unity:

$$
B_i^n(t)\ge0,
\qquad \sum_{i=0}^{n}B_i^n(t)=1.
$$

Consequently, $p(t)$ lies in the convex hull of its control values. The
package evaluates this representation using de Casteljau operations and
preserves the Bernstein basis under differentiation, integration, splitting,
and degree elevation.

Tensor-product classes replace $B_i^n$ by products of one-dimensional bases;
simplex classes use the multinomial basis described in the
[simplex guide](simplex-basis.md).
