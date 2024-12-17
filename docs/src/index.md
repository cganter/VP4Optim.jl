```@meta
CurrentModule = VP4Optim
```

# VP4Optim.jl

*Some tools to simplify the implementation of variable projection in Julia.*

## Variable Projection 

Consider a least squares (LS) cost function of the form[^1] 
```math
\chi^2\left(\bm{x}, \bm{c}\right) \;=\;
\left\|\,\bm{y} - \bm{A}(\bm{x}) \cdot \bm{c}\,\right\|^2_2
```
to be minimized with respect to ``\bm{x}`` and ``\bm{c}``
```math
\hat{\bm{x}}, \hat{\bm{c}} \;=\;
\underset{\bm{x}, \bm{c}}{\operatorname{argmin}}\;
\chi^2\left(\bm{x}, \bm{c}\right)
```
Depending on the problem of interest, data vector ``\bm{y}``, matrix ``\bm{A}`` and linear 
coefficient vector ``\bm{c}``, can be real or complex.
Only for the parameter vector ``\bm{x}``, we restrict ourselves to
real values[^2].

Variable projection ([VARPRO](https://doi.org/10.1137/0710036)) is an established method to
reduce the dimensionality of this optimization problem by eliminating the linear parameters ``\bm{c}``.
To this end, we exploit that for any given ``\bm{x}`` (not necessarily at the minimum), the minimum of 
``\chi^2\left(\bm{x}, \bm{c}\right)`` with respect to the linear coefficients ``\bm{c}`` must satisfy
```math
\bm{c}\left(\bm{x}\right) \;=\;\bm{B}^{-1}\,\bm{b}
```
where we defined[^3]
```math
\bm{B}\left(\bm{x}\right)\;:=\;\bm{A}^\ast\,\bm{A}
\qquad\qquad
\bm{b}\left(\bm{x}\right)\;:=\;\bm{A}^\ast\,\bm{y}
```
Assuming this value for ``\bm{c}``, the cost function 
```math
\chi^2\left(\bm{x}\right) \;:=\;\chi^2\left(\bm{x},\bm{c}\left(\bm{x}\right)\right)
```
depends only on ``\bm{x}`` and can be written the form[^4]:
```math
\chi^2\left(\bm{x}\right) \;=\; y^{2} \;-\; \bm{b}^\ast\,\bm{B}^{-1}\,\bm{b}
```
This allows us to first determine the estimator of the internal parameter
```math
\hat{\bm{x}} \;=\;
\underset{\bm{x}}{\operatorname{argmin}}\;
\chi^2\left(\bm{x}\right)
```
and subsequently simply calculate the optimal linear coefficient
```math
\hat{\bm{c}} \;=\; \bm{c}\left(\hat{\bm{x}}\right)
```
Note that besides reducing the number of independent parameters, the dimensions of ``\bm{B}`` and ``\bm{b}``
equal the number of elements of the linear coefficient vector ``\bm{c}``, which is often a small number.

## Package Features

The [VP4Optim](https://github.com/cganter/VP4Optim.jl) package provides some tools to simplify the implementation of VARPRO.

- template for models (real or complex) to be implemented by the user
- based upon [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) to improve performance
- wrapper for optimization libraries, such as [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
- support for partial derivatives up to second order
- test tools to check correctness of user supplied models

[^1]:
    ``\chi^2`` can result from a maximum likelihood (ML) minimization problem, except for a missing noise term. 
    In the most general cases, when the variance of the latter is not constant, such that ``y_j`` have 
    variable standard deviations ``\sigma_j``, we can recover the 
    given expression for ``\chi^2`` by simple rescaling: ``y_j/\sigma_j \to y_j`` and 
    ``A_{jk}/\sigma_j \to A_{jk}``
[^2]: 
    Most optimization libraries actually rely on this assumption, but this does not constitute a restriction:
    Any complex variable ``z = z^\prime + i\,z^{\prime\prime}`` corresponds to two real ones
    (e.g. ``z^\prime`` and ``z^{\prime\prime}``).
[^3]:
    ``\bm{A}^\ast`` denotes the conjugate tranpose of ``\bm{A}`` and we suppressed the dependence on
    ``\bm{x}`` to improve readability.    
[^4]: ``y^2 = \bm{y}^\ast\,\bm{y}``

## Manual 

```@contents
    Pages=[
        "man/example.md",
        "man/guide.md",
        "man/api.md",
        ]
    Depth=1
```

