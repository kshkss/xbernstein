# Tensor-product Bernstein polynomials

For parameter dimension $d$, degree vector
$n=(n_0,\ldots,n_{d-1})$, and multi-index
$i=(i_0,\ldots,i_{d-1})$, define

$$
B_i^n(u)=\prod_{a=0}^{d-1}B_{i_a}^{n_a}(u_a).
$$

The represented polynomial is

$$
p(u)=\sum_{i_0=0}^{n_0}\cdots\sum_{i_{d-1}=0}^{n_{d-1}}
c_iB_i^n(u).
$$

Thus a tensor polynomial stores one trailing coefficient axis per parameter.
`Bernstein2D`, `Bernstein3D`, and `Bernstein4D` use this representation.
Arithmetic broadcasts only leading batch axes; parameter axes are elevated or
aligned before arithmetic.
