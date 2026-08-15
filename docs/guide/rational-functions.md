# Rational Bernstein functions

Given control values $c_i$ and positive weights $w_i$, the rational form is

$$
N(u)=\sum_iw_ic_iB_i(u),\qquad
D(u)=\sum_iw_iB_i(u),\qquad
R(u)=\frac{N(u)}{D(u)}.
$$

Because $w_i>0$ and the Bernstein basis is nonnegative and sums to one,
$D(u)>0$ on the parameter domain. Tensor-product and simplex rational classes
use the corresponding polynomial basis. Homogeneous storage keeps $(N,D)$
together, which makes degree elevation, splitting, and differentiation
basis-preserving operations possible.
