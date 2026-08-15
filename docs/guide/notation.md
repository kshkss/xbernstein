# Mathematical notation

Throughout the documentation, $n$ denotes a polynomial degree and bold
letters denote vectors or multi-indices. A coefficient array may have leading
batch axes; the trailing axes described by a class contain its Bernstein
coefficients.

For a multi-index $\alpha=(\alpha_0,\ldots,\alpha_d)$, write

$$
|\alpha|=\sum_i\alpha_i,
\qquad
\lambda^\alpha=\prod_i\lambda_i^{\alpha_i}.
$$

Coordinates in a tensor-product domain are ordinary parameters
$u=(u_0,\ldots,u_{d-1})$. Coordinates in a simplex are barycentric
coordinates $\lambda=(\lambda_0,\ldots,\lambda_d)$ and satisfy
$\lambda_i\ge0$ and $\sum_i\lambda_i=1$.

Unless stated otherwise, a derivative axis refers to the corresponding
parameter or barycentric coordinate, and a leading array axis is preserved as
a batch of independent polynomials.
