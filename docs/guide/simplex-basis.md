# Simplex Bernstein polynomials

The standard $d$-simplex is

$$
\Delta^d=\{\lambda\in\mathbb R^{d+1}:\lambda_i\ge0,
\ \sum_{i=0}^{d}\lambda_i=1\}.
$$

For $|\alpha|=n$, its Bernstein basis is

$$
B_\alpha^n(\lambda)=
\frac{n!}{\alpha_0!\cdots\alpha_d!}
\prod_{i=0}^{d}\lambda_i^{\alpha_i}.
$$

Hence

$$
p(\lambda)=\sum_{|\alpha|=n}c_\alpha B_\alpha^n(\lambda),
\qquad
\#\{\alpha:|\alpha|=n\}=\binom{n+d}{d}.
$$

`Bernstein2DS`, `Bernstein3DS`, and `Bernstein4DS` pack these coefficients
into one final axis. Their evaluation arguments are respectively three, four,
and five barycentric coordinates. `deriv(axis=a)` is the ambient derivative
$\partial/\partial\lambda_a$; `segment(start,end)` restricts the polynomial
to $\lambda(t)=(1-t)\,start+t\,end$ and returns a one-dimensional Bernstein
polynomial.
