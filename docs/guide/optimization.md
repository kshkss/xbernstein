# Bernstein optimization

On a Bernstein domain, the polynomial value is bounded by its control values.
The minimizers use this property to subdivide the domain into boxes or
simplex regions, compute lower bounds, and retain the best candidate. Rational
functions use the corresponding dehomogenized bounds. The returned point and
value are differentiable through custom JVP rules where the active minimum is
regular.
