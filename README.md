# VP4Optim

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cganter.github.io/VP4Optim.jl/dev/)
[![Build Status](https://github.com/cganter/VP4Optim.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cganter/VP4Optim.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/cganter/VP4Optim.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cganter/VP4Optim.jl)

## Scope

For least-squares minimization problems of the form
```math
\hat{\mathbf{x}}, \hat{\mathbf{c}} \;=\;
\underset{\mathbf{x}, \mathbf{c}}{\mathop{\text{argmin}}}\;
\left\|\,\mathbf{y} - \mathbf{A}(\mathbf{x}) \cdot \mathbf{c}\,\right\|^2_2
```
variable projection ([VARPRO](https://doi.org/10.1137/0710036)) is an established technique to eliminate 
the linear coefficients $\mathbf{c}$, thereby reducing the dimensionality of the optimization space.

The VP4Optim package provides some tools to simplify the implementation of VARPRO in Julia.

## Installation

The package is not registered yet. Until then, just clone it and add it to your load path.

## Contributing

Feedback and bug reports, which adhere to the 
[Julia Community Standards](https://julialang.org/community/standards/), are welcome.

