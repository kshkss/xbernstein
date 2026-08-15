# Hermite interpolation

Hermite interpolants prescribe values and derivatives at vertices or
endpoints. For a tensor-product parameter $u$, a derivative is indexed by a
multi-index $\alpha$:

$$
\partial^\alpha p=
\frac{\partial^{|\alpha|}p}
 {\partial u_0^{\alpha_0}\cdots\partial u_{d-1}^{\alpha_{d-1}}}.
$$

The polynomial degree is selected so that the requested endpoint jet fits
the available Bernstein coefficients. Rational Hermite interpolation solves
for homogeneous numerator and denominator jets. Its staged solver initializes
vertex values, then solves independent axis subsets in increasing interaction
order: single-axis, pairwise, triplewise, and finally full mixed terms.
